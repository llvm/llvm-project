//===--- SemaAttr.cpp - Semantic Analysis for Attributes ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements semantic analysis for non-trivial attributes and
// pragmas.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/Lookup.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/Preprocessor.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// Pragma 'pack' and 'options align'
//===----------------------------------------------------------------------===//

namespace {
  struct PackStackEntry {
    // We just use a sentinel to represent when the stack is set to mac68k
    // alignment.
    static const unsigned kMac68kAlignmentSentinel = ~0U;

    unsigned Alignment;
    IdentifierInfo *Name;
  };

  /// PragmaPackStack - Simple class to wrap the stack used by #pragma
  /// pack.
  class PragmaPackStack {
    typedef std::vector<PackStackEntry> stack_ty;

    /// Alignment - The current user specified alignment.
    unsigned Alignment;

    /// Stack - Entries in the #pragma pack stack, consisting of saved
    /// alignments and optional names.
    stack_ty Stack;

  public:
    PragmaPackStack() : Alignment(0) {}

    void setAlignment(unsigned A) { Alignment = A; }
    unsigned getAlignment() { return Alignment; }

    /// push - Push the current alignment onto the stack, optionally
    /// using the given \arg Name for the record, if non-zero.
    void push(IdentifierInfo *Name) {
      PackStackEntry PSE = { Alignment, Name };
      Stack.push_back(PSE);
    }

    /// pop - Pop a record from the stack and restore the current
    /// alignment to the previous value. If \arg Name is non-zero then
    /// the first such named record is popped, otherwise the top record
    /// is popped. Returns true if the pop succeeded.
    bool pop(IdentifierInfo *Name, bool IsReset);
  };
}  // end anonymous namespace.

bool PragmaPackStack::pop(IdentifierInfo *Name, bool IsReset) {
  // If name is empty just pop top.
  if (!Name) {
    // An empty stack is a special case...
    if (Stack.empty()) {
      // If this isn't a reset, it is always an error.
      if (!IsReset)
        return false;

      // Otherwise, it is an error only if some alignment has been set.
      if (!Alignment)
        return false;

      // Otherwise, reset to the default alignment.
      Alignment = 0;
    } else {
      Alignment = Stack.back().Alignment;
      Stack.pop_back();
    }

    return true;
  }

  // Otherwise, find the named record.
  for (unsigned i = Stack.size(); i != 0; ) {
    --i;
    if (Stack[i].Name == Name) {
      // Found it, pop up to and including this record.
      Alignment = Stack[i].Alignment;
      Stack.erase(Stack.begin() + i, Stack.end());
      return true;
    }
  }

  return false;
}


/// FreePackedContext - Deallocate and null out PackContext.
void Sema::FreePackedContext() {
  delete static_cast<PragmaPackStack*>(PackContext);
  PackContext = 0;
}

void Sema::AddAlignmentAttributesForRecord(RecordDecl *RD) {
  // If there is no pack context, we don't need any attributes.
  if (!PackContext)
    return;

  PragmaPackStack *Stack = static_cast<PragmaPackStack*>(PackContext);

  // Otherwise, check to see if we need a max field alignment attribute.
  if (unsigned Alignment = Stack->getAlignment()) {
    if (Alignment == PackStackEntry::kMac68kAlignmentSentinel)
      RD->addAttr(::new (Context) AlignMac68kAttr(SourceLocation(), Context));
    else
      RD->addAttr(::new (Context) MaxFieldAlignmentAttr(SourceLocation(),
                                                        Context,
                                                        Alignment * 8));
  }
}

void Sema::AddMsStructLayoutForRecord(RecordDecl *RD) {
  if (!MSStructPragmaOn)
    return;
  RD->addAttr(::new (Context) MsStructAttr(SourceLocation(), Context));
}

void Sema::ActOnPragmaOptionsAlign(PragmaOptionsAlignKind Kind,
                                   SourceLocation PragmaLoc) {
  if (PackContext == 0)
    PackContext = new PragmaPackStack();

  PragmaPackStack *Context = static_cast<PragmaPackStack*>(PackContext);

  switch (Kind) {
    // For all targets we support native and natural are the same.
    //
    // FIXME: This is not true on Darwin/PPC.
  case POAK_Native:
  case POAK_Power:
  case POAK_Natural:
    Context->push(0);
    Context->setAlignment(0);
    break;

    // Note that '#pragma options align=packed' is not equivalent to attribute
    // packed, it has a different precedence relative to attribute aligned.
  case POAK_Packed:
    Context->push(0);
    Context->setAlignment(1);
    break;

  case POAK_Mac68k:
    // Check if the target supports this.
    if (!PP.getTargetInfo().hasAlignMac68kSupport()) {
      Diag(PragmaLoc, diag::err_pragma_options_align_mac68k_target_unsupported);
      return;
    }
    Context->push(0);
    Context->setAlignment(PackStackEntry::kMac68kAlignmentSentinel);
    break;

  case POAK_Reset:
    // Reset just pops the top of the stack, or resets the current alignment to
    // default.
    if (!Context->pop(0, /*IsReset=*/true)) {
      Diag(PragmaLoc, diag::warn_pragma_options_align_reset_failed)
        << "stack empty";
    }
    break;
  }
}

void Sema::ActOnPragmaPack(PragmaPackKind Kind, IdentifierInfo *Name,
                           Expr *alignment, SourceLocation PragmaLoc,
                           SourceLocation LParenLoc, SourceLocation RParenLoc) {
  Expr *Alignment = static_cast<Expr *>(alignment);

  // If specified then alignment must be a "small" power of two.
  unsigned AlignmentVal = 0;
  if (Alignment) {
    llvm::APSInt Val;

    // pack(0) is like pack(), which just works out since that is what
    // we use 0 for in PackAttr.
    if (Alignment->isTypeDependent() ||
        Alignment->isValueDependent() ||
        !Alignment->isIntegerConstantExpr(Val, Context) ||
        !(Val == 0 || Val.isPowerOf2()) ||
        Val.getZExtValue() > 16) {
      Diag(PragmaLoc, diag::warn_pragma_pack_invalid_alignment);
      return; // Ignore
    }

    AlignmentVal = (unsigned) Val.getZExtValue();
  }

  if (PackContext == 0)
    PackContext = new PragmaPackStack();

  PragmaPackStack *Context = static_cast<PragmaPackStack*>(PackContext);

  switch (Kind) {
  case Sema::PPK_Default: // pack([n])
    Context->setAlignment(AlignmentVal);
    break;

  case Sema::PPK_Show: // pack(show)
    // Show the current alignment, making sure to show the right value
    // for the default.
    AlignmentVal = Context->getAlignment();
    // FIXME: This should come from the target.
    if (AlignmentVal == 0)
      AlignmentVal = 8;
    if (AlignmentVal == PackStackEntry::kMac68kAlignmentSentinel)
      Diag(PragmaLoc, diag::warn_pragma_pack_show) << "mac68k";
    else
      Diag(PragmaLoc, diag::warn_pragma_pack_show) << AlignmentVal;
    break;

  case Sema::PPK_Push: // pack(push [, id] [, [n])
    Context->push(Name);
    // Set the new alignment if specified.
    if (Alignment)
      Context->setAlignment(AlignmentVal);
    break;

  case Sema::PPK_Pop: // pack(pop [, id] [,  n])
    // MSDN, C/C++ Preprocessor Reference > Pragma Directives > pack:
    // "#pragma pack(pop, identifier, n) is undefined"
    if (Alignment && Name)
      Diag(PragmaLoc, diag::warn_pragma_pack_pop_identifer_and_alignment);

    // Do the pop.
    if (!Context->pop(Name, /*IsReset=*/false)) {
      // If a name was specified then failure indicates the name
      // wasn't found. Otherwise failure indicates the stack was
      // empty.
      Diag(PragmaLoc, diag::warn_pragma_pack_pop_failed)
        << (Name ? "no record matching name" : "stack empty");

      // FIXME: Warn about popping named records as MSVC does.
    } else {
      // Pop succeeded, set the new alignment if specified.
      if (Alignment)
        Context->setAlignment(AlignmentVal);
    }
    break;
  }
}

void Sema::ActOnPragmaMSStruct(PragmaMSStructKind Kind) { 
  MSStructPragmaOn = (Kind == PMSST_ON);
}

void Sema::ActOnPragmaUnused(const Token &IdTok, Scope *curScope,
                             SourceLocation PragmaLoc) {

  IdentifierInfo *Name = IdTok.getIdentifierInfo();
  LookupResult Lookup(*this, Name, IdTok.getLocation(), LookupOrdinaryName);
  LookupParsedName(Lookup, curScope, NULL, true);

  if (Lookup.empty()) {
    Diag(PragmaLoc, diag::warn_pragma_unused_undeclared_var)
      << Name << SourceRange(IdTok.getLocation());
    return;
  }

  VarDecl *VD = Lookup.getAsSingle<VarDecl>();
  if (!VD) {
    Diag(PragmaLoc, diag::warn_pragma_unused_expected_var_arg)
      << Name << SourceRange(IdTok.getLocation());
    return;
  }

  // Warn if this was used before being marked unused.
  if (VD->isUsed())
    Diag(PragmaLoc, diag::warn_used_but_marked_unused) << Name;

  VD->addAttr(::new (Context) UnusedAttr(IdTok.getLocation(), Context));
}

void Sema::AddCFAuditedAttribute(Decl *D) {
  SourceLocation Loc = PP.getPragmaARCCFCodeAuditedLoc();
  if (!Loc.isValid()) return;

  // Don't add a redundant or conflicting attribute.
  if (D->hasAttr<CFAuditedTransferAttr>() ||
      D->hasAttr<CFUnknownTransferAttr>())
    return;

  D->addAttr(::new (Context) CFAuditedTransferAttr(Loc, Context));
}

typedef std::vector<std::pair<unsigned, SourceLocation> > VisStack;
enum { NoVisibility = (unsigned) -1 };

void Sema::AddPushedVisibilityAttribute(Decl *D) {
  if (!VisContext)
    return;

  if (isa<NamedDecl>(D) && cast<NamedDecl>(D)->getExplicitVisibility())
    return;

  VisStack *Stack = static_cast<VisStack*>(VisContext);
  unsigned rawType = Stack->back().first;
  if (rawType == NoVisibility) return;

  VisibilityAttr::VisibilityType type
    = (VisibilityAttr::VisibilityType) rawType;
  SourceLocation loc = Stack->back().second;

  D->addAttr(::new (Context) VisibilityAttr(loc, Context, type));
}

/// FreeVisContext - Deallocate and null out VisContext.
void Sema::FreeVisContext() {
  delete static_cast<VisStack*>(VisContext);
  VisContext = 0;
}

static void PushPragmaVisibility(Sema &S, unsigned type, SourceLocation loc) {
  // Put visibility on stack.
  if (!S.VisContext)
    S.VisContext = new VisStack;

  VisStack *Stack = static_cast<VisStack*>(S.VisContext);
  Stack->push_back(std::make_pair(type, loc));
}

void Sema::ActOnPragmaVisibility(const IdentifierInfo* VisType,
                                 SourceLocation PragmaLoc) {
  if (VisType) {
    // Compute visibility to use.
    VisibilityAttr::VisibilityType type;
    if (VisType->isStr("default"))
      type = VisibilityAttr::Default;
    else if (VisType->isStr("hidden"))
      type = VisibilityAttr::Hidden;
    else if (VisType->isStr("internal"))
      type = VisibilityAttr::Hidden; // FIXME
    else if (VisType->isStr("protected"))
      type = VisibilityAttr::Protected;
    else {
      Diag(PragmaLoc, diag::warn_attribute_unknown_visibility) <<
        VisType->getName();
      return;
    }
    PushPragmaVisibility(*this, type, PragmaLoc);
  } else {
    PopPragmaVisibility(false, PragmaLoc);
  }
}

void Sema::ActOnPragmaFPContract(tok::OnOffSwitch OOS) {
  switch (OOS) {
  case tok::OOS_ON:
    FPFeatures.fp_contract = 1;
    break;
  case tok::OOS_OFF:
    FPFeatures.fp_contract = 0; 
    break;
  case tok::OOS_DEFAULT:
    FPFeatures.fp_contract = getLangOpts().DefaultFPContract;
    break;
  }
}

void Sema::PushNamespaceVisibilityAttr(const VisibilityAttr *Attr,
                                       SourceLocation Loc) {
  // Visibility calculations will consider the namespace's visibility.
  // Here we just want to note that we're in a visibility context
  // which overrides any enclosing #pragma context, but doesn't itself
  // contribute visibility.
  PushPragmaVisibility(*this, NoVisibility, Loc);
}

void Sema::PopPragmaVisibility(bool IsNamespaceEnd, SourceLocation EndLoc) {
  if (!VisContext) {
    Diag(EndLoc, diag::err_pragma_pop_visibility_mismatch);
    return;
  }

  // Pop visibility from stack
  VisStack *Stack = static_cast<VisStack*>(VisContext);

  const std::pair<unsigned, SourceLocation> *Back = &Stack->back();
  bool StartsWithPragma = Back->first != NoVisibility;
  if (StartsWithPragma && IsNamespaceEnd) {
    Diag(Back->second, diag::err_pragma_push_visibility_mismatch);
    Diag(EndLoc, diag::note_surrounding_namespace_ends_here);

    // For better error recovery, eat all pushes inside the namespace.
    do {
      Stack->pop_back();
      Back = &Stack->back();
      StartsWithPragma = Back->first != NoVisibility;
    } while (StartsWithPragma);
  } else if (!StartsWithPragma && !IsNamespaceEnd) {
    Diag(EndLoc, diag::err_pragma_pop_visibility_mismatch);
    Diag(Back->second, diag::note_surrounding_namespace_starts_here);
    return;
  }

  Stack->pop_back();
  // To simplify the implementation, never keep around an empty stack.
  if (Stack->empty())
    FreeVisContext();
}
