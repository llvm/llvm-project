//===--- SemaExprObjC.cpp - Semantic Analysis for ObjC Expressions --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for Objective-C expressions.
//
//===----------------------------------------------------------------------===//

#include "Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/ExprObjC.h"
#include "llvm/ADT/SmallString.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang;

Sema::ExprResult Sema::ParseObjCStringLiteral(SourceLocation *AtLocs, 
                                              ExprTy **strings,
                                              unsigned NumStrings) {
  StringLiteral **Strings = reinterpret_cast<StringLiteral**>(strings);

  // Most ObjC strings are formed out of a single piece.  However, we *can*
  // have strings formed out of multiple @ strings with multiple pptokens in
  // each one, e.g. @"foo" "bar" @"baz" "qux"   which need to be turned into one
  // StringLiteral for ObjCStringLiteral to hold onto.
  StringLiteral *S = Strings[0];
  
  // If we have a multi-part string, merge it all together.
  if (NumStrings != 1) {
    // Concatenate objc strings.
    llvm::SmallString<128> StrBuf;
    llvm::SmallVector<SourceLocation, 8> StrLocs;
    
    for (unsigned i = 0; i != NumStrings; ++i) {
      S = Strings[i];
      
      // ObjC strings can't be wide.
      if (S->isWide()) {
        Diag(S->getLocStart(), diag::err_cfstring_literal_not_string_constant)
          << S->getSourceRange();
        return true;
      }
      
      // Get the string data.
      StrBuf.append(S->getStrData(), S->getStrData()+S->getByteLength());
      
      // Get the locations of the string tokens.
      StrLocs.append(S->tokloc_begin(), S->tokloc_end());
      
      // Free the temporary string.
      S->Destroy(Context);
    }
    
    // Create the aggregate string with the appropriate content and location
    // information.
    S = StringLiteral::Create(Context, &StrBuf[0], StrBuf.size(), false,
                              Context.getPointerType(Context.CharTy),
                              &StrLocs[0], StrLocs.size());
  }
  
  // Verify that this composite string is acceptable for ObjC strings.
  if (CheckObjCString(S))
    return true;

  // Initialize the constant string interface lazily. This assumes
  // the NSString interface is seen in this translation unit. Note: We
  // don't use NSConstantString, since the runtime team considers this
  // interface private (even though it appears in the header files).
  QualType Ty = Context.getObjCConstantStringInterface();
  if (!Ty.isNull()) {
    Ty = Context.getObjCObjectPointerType(Ty);
  } else {
    IdentifierInfo *NSIdent = &Context.Idents.get("NSString");
    NamedDecl *IF = LookupName(TUScope, NSIdent, LookupOrdinaryName);
    if (ObjCInterfaceDecl *StrIF = dyn_cast_or_null<ObjCInterfaceDecl>(IF)) {
      Context.setObjCConstantStringInterface(StrIF);
      Ty = Context.getObjCConstantStringInterface();
      Ty = Context.getObjCObjectPointerType(Ty);
    } else {
      // If there is no NSString interface defined then treat constant
      // strings as untyped objects and let the runtime figure it out later.
      Ty = Context.getObjCIdType();
    }
  }
  
  return new (Context) ObjCStringLiteral(S, Ty, AtLocs[0]);
}

Expr *Sema::BuildObjCEncodeExpression(SourceLocation AtLoc, 
                                      QualType EncodedType,
                                      SourceLocation RParenLoc) {
  QualType StrTy;
  if (EncodedType->isDependentType()) 
    StrTy = Context.DependentTy;
  else {
    std::string Str;
    Context.getObjCEncodingForType(EncodedType, Str);

    // The type of @encode is the same as the type of the corresponding string,
    // which is an array type.
    StrTy = Context.CharTy;
    // A C++ string literal has a const-qualified element type (C++ 2.13.4p1).
    if (getLangOptions().CPlusPlus)
      StrTy.addConst();
    StrTy = Context.getConstantArrayType(StrTy, llvm::APInt(32, Str.size()+1),
                                         ArrayType::Normal, 0);
  }
  
  return new (Context) ObjCEncodeExpr(StrTy, EncodedType, AtLoc, RParenLoc);
}

Sema::ExprResult Sema::ParseObjCEncodeExpression(SourceLocation AtLoc,
                                                 SourceLocation EncodeLoc,
                                                 SourceLocation LParenLoc,
                                                 TypeTy *ty,
                                                 SourceLocation RParenLoc) {
  // FIXME: Preserve type source info ?
  QualType EncodedType = GetTypeFromParser(ty);

  return BuildObjCEncodeExpression(AtLoc, EncodedType, RParenLoc);
}

Sema::ExprResult Sema::ParseObjCSelectorExpression(Selector Sel,
                                                   SourceLocation AtLoc,
                                                   SourceLocation SelLoc,
                                                   SourceLocation LParenLoc,
                                                   SourceLocation RParenLoc) {
  ObjCMethodDecl *Method = LookupInstanceMethodInGlobalPool(Sel, 
                             SourceRange(LParenLoc, RParenLoc), false);
  if (!Method)
    Method = LookupFactoryMethodInGlobalPool(Sel,
                                          SourceRange(LParenLoc, RParenLoc));
  if (!Method)
    Diag(SelLoc, diag::warn_undeclared_selector) << Sel;

  QualType Ty = Context.getObjCSelType();
  return new (Context) ObjCSelectorExpr(Ty, Sel, AtLoc, RParenLoc);
}

Sema::ExprResult Sema::ParseObjCProtocolExpression(IdentifierInfo *ProtocolId,
                                                   SourceLocation AtLoc,
                                                   SourceLocation ProtoLoc,
                                                   SourceLocation LParenLoc,
                                                   SourceLocation RParenLoc) {
  ObjCProtocolDecl* PDecl = LookupProtocol(ProtocolId);
  if (!PDecl) {
    Diag(ProtoLoc, diag::err_undeclared_protocol) << ProtocolId;
    return true;
  }
  
  QualType Ty = Context.getObjCProtoType();
  if (Ty.isNull())
    return true;
  Ty = Context.getObjCObjectPointerType(Ty);
  return new (Context) ObjCProtocolExpr(Ty, PDecl, AtLoc, RParenLoc);
}

bool Sema::CheckMessageArgumentTypes(Expr **Args, unsigned NumArgs, 
                                     Selector Sel, ObjCMethodDecl *Method, 
                                     bool isClassMessage,
                                     SourceLocation lbrac, SourceLocation rbrac,
                                     QualType &ReturnType) {  
  if (!Method) {
    // Apply default argument promotion as for (C99 6.5.2.2p6).
    for (unsigned i = 0; i != NumArgs; i++)
      DefaultArgumentPromotion(Args[i]);

    unsigned DiagID = isClassMessage ? diag::warn_class_method_not_found :
                                       diag::warn_inst_method_not_found;
    Diag(lbrac, DiagID)
      << Sel << isClassMessage << SourceRange(lbrac, rbrac);
    ReturnType = Context.getObjCIdType();
    return false;
  }
  
  ReturnType = Method->getResultType();
   
  unsigned NumNamedArgs = Sel.getNumArgs();
  assert(NumArgs >= NumNamedArgs && "Too few arguments for selector!");

  bool IsError = false;
  for (unsigned i = 0; i < NumNamedArgs; i++) {
    Expr *argExpr = Args[i];
    assert(argExpr && "CheckMessageArgumentTypes(): missing expression");
    
    QualType lhsType = Method->param_begin()[i]->getType();
    QualType rhsType = argExpr->getType();

    // If necessary, apply function/array conversion. C99 6.7.5.3p[7,8]. 
    if (lhsType->isArrayType())
      lhsType = Context.getArrayDecayedType(lhsType);
    else if (lhsType->isFunctionType())
      lhsType = Context.getPointerType(lhsType);

    AssignConvertType Result = 
      CheckSingleAssignmentConstraints(lhsType, argExpr);
    if (Args[i] != argExpr) // The expression was converted.
      Args[i] = argExpr; // Make sure we store the converted expression.
    
    IsError |= 
      DiagnoseAssignmentResult(Result, argExpr->getLocStart(), lhsType, rhsType,
                               argExpr, "sending");
  }

  // Promote additional arguments to variadic methods.
  if (Method->isVariadic()) {
    for (unsigned i = NumNamedArgs; i < NumArgs; ++i)
      IsError |= DefaultVariadicArgumentPromotion(Args[i], VariadicMethod);
  } else {
    // Check for extra arguments to non-variadic methods.
    if (NumArgs != NumNamedArgs) {
      Diag(Args[NumNamedArgs]->getLocStart(), 
           diag::err_typecheck_call_too_many_args)
        << 2 /*method*/ << Method->getSourceRange()
        << SourceRange(Args[NumNamedArgs]->getLocStart(),
                       Args[NumArgs-1]->getLocEnd());
    }
  }

  return IsError;
}

bool Sema::isSelfExpr(Expr *RExpr) {
  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(RExpr))
    if (DRE->getDecl()->getIdentifier() == &Context.Idents.get("self"))
      return true;
  return false;
}

// Helper method for ActOnClassMethod/ActOnInstanceMethod.
// Will search "local" class/category implementations for a method decl.
// If failed, then we search in class's root for an instance method.
// Returns 0 if no method is found.
ObjCMethodDecl *Sema::LookupPrivateClassMethod(Selector Sel,
                                          ObjCInterfaceDecl *ClassDecl) {
  ObjCMethodDecl *Method = 0;
  // lookup in class and all superclasses
  while (ClassDecl && !Method) {
    if (ObjCImplementationDecl *ImpDecl = ClassDecl->getImplementation())
      Method = ImpDecl->getClassMethod(Sel);
    
    // Look through local category implementations associated with the class.
    if (!Method)
      Method = ClassDecl->getCategoryClassMethod(Sel);
    
    // Before we give up, check if the selector is an instance method.
    // But only in the root. This matches gcc's behaviour and what the
    // runtime expects.
    if (!Method && !ClassDecl->getSuperClass()) {
      Method = ClassDecl->lookupInstanceMethod(Sel);
      // Look through local category implementations associated 
      // with the root class.
      if (!Method) 
        Method = LookupPrivateInstanceMethod(Sel, ClassDecl);
    }
    
    ClassDecl = ClassDecl->getSuperClass();
  }
  return Method;
}

ObjCMethodDecl *Sema::LookupPrivateInstanceMethod(Selector Sel,
                                              ObjCInterfaceDecl *ClassDecl) {
  ObjCMethodDecl *Method = 0;
  while (ClassDecl && !Method) {
    // If we have implementations in scope, check "private" methods.
    if (ObjCImplementationDecl *ImpDecl = ClassDecl->getImplementation())
      Method = ImpDecl->getInstanceMethod(Sel);
    
    // Look through local category implementations associated with the class.
    if (!Method)
      Method = ClassDecl->getCategoryInstanceMethod(Sel);
    ClassDecl = ClassDecl->getSuperClass();
  }
  return Method;
}

Action::OwningExprResult Sema::ActOnClassPropertyRefExpr(
  IdentifierInfo &receiverName,
  IdentifierInfo &propertyName,
  SourceLocation &receiverNameLoc,
  SourceLocation &propertyNameLoc) {
  
  ObjCInterfaceDecl *IFace = getObjCInterfaceDecl(&receiverName);
  
  // Search for a declared property first.
  
  Selector Sel = PP.getSelectorTable().getNullarySelector(&propertyName);
  ObjCMethodDecl *Getter = IFace->lookupClassMethod(Sel);

  // If this reference is in an @implementation, check for 'private' methods.
  if (!Getter)
    if (ObjCMethodDecl *CurMeth = getCurMethodDecl())
      if (ObjCInterfaceDecl *ClassDecl = CurMeth->getClassInterface())
        if (ObjCImplementationDecl *ImpDecl = ClassDecl->getImplementation())
          Getter = ImpDecl->getClassMethod(Sel);

  if (Getter) {
    // FIXME: refactor/share with ActOnMemberReference().
    // Check if we can reference this property.
    if (DiagnoseUseOfDecl(Getter, propertyNameLoc))
      return ExprError();
  }
  
  // Look for the matching setter, in case it is needed.
  Selector SetterSel = 
    SelectorTable::constructSetterName(PP.getIdentifierTable(), 
                                       PP.getSelectorTable(), &propertyName);
    
  ObjCMethodDecl *Setter = IFace->lookupClassMethod(SetterSel);
  if (!Setter) {
    // If this reference is in an @implementation, also check for 'private'
    // methods.
    if (ObjCMethodDecl *CurMeth = getCurMethodDecl())
      if (ObjCInterfaceDecl *ClassDecl = CurMeth->getClassInterface())
        if (ObjCImplementationDecl *ImpDecl = ClassDecl->getImplementation())
          Setter = ImpDecl->getClassMethod(SetterSel);
  }
  // Look through local category implementations associated with the class.
  if (!Setter)
    Setter = IFace->getCategoryClassMethod(SetterSel);

  if (Setter && DiagnoseUseOfDecl(Setter, propertyNameLoc))
    return ExprError();

  if (Getter || Setter) {
    QualType PType;

    if (Getter)
      PType = Getter->getResultType();
    else {
      for (ObjCMethodDecl::param_iterator PI = Setter->param_begin(),
           E = Setter->param_end(); PI != E; ++PI)
        PType = (*PI)->getType();
    }
    return Owned(new (Context) ObjCImplicitSetterGetterRefExpr(
                                  Getter, PType, Setter, 
                                  propertyNameLoc, IFace, receiverNameLoc));
  }
  return ExprError(Diag(propertyNameLoc, diag::err_property_not_found)
                     << &propertyName << Context.getObjCInterfaceType(IFace));
}


// ActOnClassMessage - used for both unary and keyword messages.
// ArgExprs is optional - if it is present, the number of expressions
// is obtained from Sel.getNumArgs().
Sema::ExprResult Sema::ActOnClassMessage(
  Scope *S,
  IdentifierInfo *receiverName, Selector Sel,
  SourceLocation lbrac, SourceLocation receiverLoc,
  SourceLocation selectorLoc, SourceLocation rbrac, 
  ExprTy **Args, unsigned NumArgs)
{
  assert(receiverName && "missing receiver class name");

  Expr **ArgExprs = reinterpret_cast<Expr **>(Args);
  ObjCInterfaceDecl* ClassDecl = 0;
  bool isSuper = false;
  
  if (receiverName->isStr("super")) {
    if (getCurMethodDecl()) {
      isSuper = true;
      ObjCInterfaceDecl *OID = getCurMethodDecl()->getClassInterface();
      if (!OID)
        return Diag(lbrac, diag::error_no_super_class_message) 
                      << getCurMethodDecl()->getDeclName();
      ClassDecl = OID->getSuperClass();
      if (!ClassDecl)
        return Diag(lbrac, diag::error_no_super_class) << OID->getDeclName();
      if (getCurMethodDecl()->isInstanceMethod()) {
        QualType superTy = Context.getObjCInterfaceType(ClassDecl);
        superTy = Context.getObjCObjectPointerType(superTy);
        ExprResult ReceiverExpr = new (Context) ObjCSuperExpr(SourceLocation(),
                                                              superTy);
        // We are really in an instance method, redirect.
        return ActOnInstanceMessage(ReceiverExpr.get(), Sel, lbrac, 
                                    selectorLoc, rbrac, Args, NumArgs);
      }
      // We are sending a message to 'super' within a class method. Do nothing,
      // the receiver will pass through as 'super' (how convenient:-).
    } else {
      // 'super' has been used outside a method context. If a variable named
      // 'super' has been declared, redirect. If not, produce a diagnostic.
      NamedDecl *SuperDecl = LookupName(S, receiverName, LookupOrdinaryName);
      ValueDecl *VD = dyn_cast_or_null<ValueDecl>(SuperDecl);
      if (VD) {
        ExprResult ReceiverExpr = new (Context) DeclRefExpr(VD, VD->getType(), 
                                                            receiverLoc);
        // We are really in an instance method, redirect.
        return ActOnInstanceMessage(ReceiverExpr.get(), Sel, lbrac, 
                                    selectorLoc, rbrac, Args, NumArgs);
      }
      return Diag(receiverLoc, diag::err_undeclared_var_use) << receiverName;
    }      
  } else
    ClassDecl = getObjCInterfaceDecl(receiverName);
  
  // The following code allows for the following GCC-ism:
  //
  //  typedef XCElementDisplayRect XCElementGraphicsRect;
  //
  //  @implementation XCRASlice
  //  - whatever { // Note that XCElementGraphicsRect is a typedef name.
  //    _sGraphicsDelegate =[[XCElementGraphicsRect alloc] init];
  //  }
  //
  // If necessary, the following lookup could move to getObjCInterfaceDecl().
  if (!ClassDecl) {
    NamedDecl *IDecl = LookupName(TUScope, receiverName, LookupOrdinaryName);
    if (TypedefDecl *OCTD = dyn_cast_or_null<TypedefDecl>(IDecl)) {
      const ObjCInterfaceType *OCIT;
      OCIT = OCTD->getUnderlyingType()->getAsObjCInterfaceType();
      if (!OCIT) {
        Diag(receiverLoc, diag::err_invalid_receiver_to_message);
        return true;
      }
      ClassDecl = OCIT->getDecl();
    }
  }
  assert(ClassDecl && "missing interface declaration");
  ObjCMethodDecl *Method = 0;
  QualType returnType;
  if (ClassDecl->isForwardDecl()) {
    // A forward class used in messaging is tread as a 'Class'
    Diag(lbrac, diag::warn_receiver_forward_class) << ClassDecl->getDeclName();
    Method = LookupFactoryMethodInGlobalPool(Sel, SourceRange(lbrac,rbrac));
    if (Method)
      Diag(Method->getLocation(), diag::note_method_sent_forward_class) 
        << Method->getDeclName();
  }
  if (!Method)
    Method = ClassDecl->lookupClassMethod(Sel);
  
  // If we have an implementation in scope, check "private" methods.
  if (!Method)
    Method = LookupPrivateClassMethod(Sel, ClassDecl);

  if (Method && DiagnoseUseOfDecl(Method, receiverLoc))
    return true;
  
  if (CheckMessageArgumentTypes(ArgExprs, NumArgs, Sel, Method, true, 
                                lbrac, rbrac, returnType))
    return true;

  returnType = returnType.getNonReferenceType();
  
  // If we have the ObjCInterfaceDecl* for the class that is receiving the
  // message, use that to construct the ObjCMessageExpr.  Otherwise pass on the
  // IdentifierInfo* for the class.
  // FIXME: need to do a better job handling 'super' usage within a class.  For
  // now, we simply pass the "super" identifier through (which isn't consistent
  // with instance methods.
  if (isSuper)
    return new (Context) ObjCMessageExpr(receiverName, Sel, returnType, Method,
                                         lbrac, rbrac, ArgExprs, NumArgs);
  else
    return new (Context) ObjCMessageExpr(ClassDecl, Sel, returnType, Method,
                                         lbrac, rbrac, ArgExprs, NumArgs);
}

// ActOnInstanceMessage - used for both unary and keyword messages.
// ArgExprs is optional - if it is present, the number of expressions
// is obtained from Sel.getNumArgs().
Sema::ExprResult Sema::ActOnInstanceMessage(ExprTy *receiver, Selector Sel,
                                            SourceLocation lbrac, 
                                            SourceLocation receiverLoc,
                                            SourceLocation rbrac,
                                            ExprTy **Args, unsigned NumArgs) {
  assert(receiver && "missing receiver expression");
  
  Expr **ArgExprs = reinterpret_cast<Expr **>(Args);
  Expr *RExpr = static_cast<Expr *>(receiver);
  
  // If necessary, apply function/array conversion to the receiver.
  // C99 6.7.5.3p[7,8].
  DefaultFunctionArrayConversion(RExpr);
  
  QualType returnType;
  QualType ReceiverCType =
    Context.getCanonicalType(RExpr->getType()).getUnqualifiedType();

  // Handle messages to 'super'.
  if (isa<ObjCSuperExpr>(RExpr)) {
    ObjCMethodDecl *Method = 0;
    if (ObjCMethodDecl *CurMeth = getCurMethodDecl()) {
      // If we have an interface in scope, check 'super' methods.
      if (ObjCInterfaceDecl *ClassDecl = CurMeth->getClassInterface())
        if (ObjCInterfaceDecl *SuperDecl = ClassDecl->getSuperClass()) {
          Method = SuperDecl->lookupInstanceMethod(Sel);
          
          if (!Method) 
            // If we have implementations in scope, check "private" methods.
            Method = LookupPrivateInstanceMethod(Sel, SuperDecl);
        }
    }

    if (Method && DiagnoseUseOfDecl(Method, receiverLoc))
      return true;

    if (CheckMessageArgumentTypes(ArgExprs, NumArgs, Sel, Method, false,
                                  lbrac, rbrac, returnType))
      return true;
    
    returnType = returnType.getNonReferenceType();
    return new (Context) ObjCMessageExpr(RExpr, Sel, returnType, Method, lbrac,
                                         rbrac, ArgExprs, NumArgs);
  }

  // Handle messages to id.  
  if (ReceiverCType->isObjCIdType() || ReceiverCType->isBlockPointerType() ||
      Context.isObjCNSObjectType(RExpr->getType())) {
    ObjCMethodDecl *Method = LookupInstanceMethodInGlobalPool(
                               Sel, SourceRange(lbrac,rbrac));
    if (!Method)
      Method = LookupFactoryMethodInGlobalPool(Sel, SourceRange(lbrac, rbrac));
    if (CheckMessageArgumentTypes(ArgExprs, NumArgs, Sel, Method, false, 
                                  lbrac, rbrac, returnType))
      return true;
    returnType = returnType.getNonReferenceType();
    return new (Context) ObjCMessageExpr(RExpr, Sel, returnType, Method, lbrac,
                                         rbrac, ArgExprs, NumArgs);
  }
  
  // Handle messages to Class.
  if (ReceiverCType->isObjCClassType() || 
      ReceiverCType->isObjCQualifiedClassType()) {
    ObjCMethodDecl *Method = 0;
    
    if (ObjCMethodDecl *CurMeth = getCurMethodDecl()) {
      if (ObjCInterfaceDecl *ClassDecl = CurMeth->getClassInterface()) {
        // First check the public methods in the class interface.
        Method = ClassDecl->lookupClassMethod(Sel);
        
        if (!Method)
          Method = LookupPrivateClassMethod(Sel, ClassDecl);
          
        // FIXME: if we still haven't found a method, we need to look in 
        // protocols (if we have qualifiers).
      }
      if (Method && DiagnoseUseOfDecl(Method, receiverLoc))
        return true;
    }
    if (!Method) {
      // If not messaging 'self', look for any factory method named 'Sel'.
      if (!isSelfExpr(RExpr)) {
        Method = LookupFactoryMethodInGlobalPool(Sel, SourceRange(lbrac,rbrac));
        if (!Method) {
          // If no class (factory) method was found, check if an _instance_
          // method of the same name exists in the root class only.
          Method = LookupInstanceMethodInGlobalPool(
                                   Sel, SourceRange(lbrac,rbrac));
          if (Method)
              if (const ObjCInterfaceDecl *ID =
                dyn_cast<ObjCInterfaceDecl>(Method->getDeclContext())) {
              if (ID->getSuperClass())
                Diag(lbrac, diag::warn_root_inst_method_not_found)
                  << Sel << SourceRange(lbrac, rbrac);
            }
        }
      }
    }
    if (CheckMessageArgumentTypes(ArgExprs, NumArgs, Sel, Method, false,
                                  lbrac, rbrac, returnType))
      return true;
    returnType = returnType.getNonReferenceType();
    return new (Context) ObjCMessageExpr(RExpr, Sel, returnType, Method, lbrac,
                                         rbrac, ArgExprs, NumArgs);
  }
  
  ObjCMethodDecl *Method = 0;
  ObjCInterfaceDecl* ClassDecl = 0;
  
  // We allow sending a message to a qualified ID ("id<foo>"), which is ok as 
  // long as one of the protocols implements the selector (if not, warn).
  if (const ObjCObjectPointerType *QIdTy = 
        ReceiverCType->getAsObjCQualifiedIdType()) {
    // Search protocols for instance methods.
    for (ObjCObjectPointerType::qual_iterator I = QIdTy->qual_begin(),
         E = QIdTy->qual_end(); I != E; ++I) {
      ObjCProtocolDecl *PDecl = *I;
      if (PDecl && (Method = PDecl->lookupInstanceMethod(Sel)))
        break;
      // Since we aren't supporting "Class<foo>", look for a class method.
      if (PDecl && (Method = PDecl->lookupClassMethod(Sel)))
        break;
    }
  } else if (const ObjCObjectPointerType *OCIType = 
                ReceiverCType->getAsObjCInterfacePointerType()) {
    // We allow sending a message to a pointer to an interface (an object).
    
    ClassDecl = OCIType->getInterfaceDecl();
    // FIXME: consider using LookupInstanceMethodInGlobalPool, since it will be
    // faster than the following method (which can do *many* linear searches).
    // The idea is to add class info to InstanceMethodPool.
    Method = ClassDecl->lookupInstanceMethod(Sel);
    
    if (!Method) {
      // Search protocol qualifiers.
      for (ObjCObjectPointerType::qual_iterator QI = OCIType->qual_begin(),
           E = OCIType->qual_end(); QI != E; ++QI) {
        if ((Method = (*QI)->lookupInstanceMethod(Sel)))
          break;
      }
    }
    if (!Method) {
      // If we have implementations in scope, check "private" methods.
      Method = LookupPrivateInstanceMethod(Sel, ClassDecl);
      
      if (!Method && !isSelfExpr(RExpr)) {
        // If we still haven't found a method, look in the global pool. This
        // behavior isn't very desirable, however we need it for GCC
        // compatibility. FIXME: should we deviate??
        if (OCIType->qual_empty()) {
          Method = LookupInstanceMethodInGlobalPool(
                               Sel, SourceRange(lbrac,rbrac));
          if (Method && !OCIType->getInterfaceDecl()->isForwardDecl())
            Diag(lbrac, diag::warn_maynot_respond) 
              << OCIType->getInterfaceDecl()->getIdentifier()->getName() << Sel;
        }
      }
    }
    if (Method && DiagnoseUseOfDecl(Method, receiverLoc))
      return true;
  } else if (!Context.getObjCIdType().isNull() &&
             (ReceiverCType->isPointerType() ||
              (ReceiverCType->isIntegerType() && 
               ReceiverCType->isScalarType()))) {
    // Implicitly convert integers and pointers to 'id' but emit a warning.
    Diag(lbrac, diag::warn_bad_receiver_type)
      << RExpr->getType() << RExpr->getSourceRange();
    ImpCastExprToType(RExpr, Context.getObjCIdType());
  } else {
    // Reject other random receiver types (e.g. structs).
    Diag(lbrac, diag::err_bad_receiver_type)
      << RExpr->getType() << RExpr->getSourceRange();
    return true;
  }
  
  if (Method)
    DiagnoseSentinelCalls(Method, receiverLoc, ArgExprs, NumArgs);
  if (CheckMessageArgumentTypes(ArgExprs, NumArgs, Sel, Method, false,
                                lbrac, rbrac, returnType))
    return true;
  returnType = returnType.getNonReferenceType();
  return new (Context) ObjCMessageExpr(RExpr, Sel, returnType, Method, lbrac,
                                       rbrac, ArgExprs, NumArgs);
}

