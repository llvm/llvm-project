//===-- lib/Parser/unparse.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Generates Fortran from the content of a parse tree, using the
// traversal templates in parse-tree-visitor.h.

#include "flang/Parser/unparse.h"
#include "flang/Common/idioms.h"
#include "flang/Common/indirection.h"
#include "flang/Parser/characters.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Parser/tools.h"
#include "flang/Support/Fortran.h"
#include "flang/Support/LangOptions.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cinttypes>
#include <cstddef>
#include <set>

namespace Fortran::parser {

class UnparseVisitor {
public:
  UnparseVisitor(llvm::raw_ostream &out, const common::LangOptions &langOpts,
      int indentationAmount, Encoding encoding, bool capitalize,
      bool backslashEscapes, preStatementType *preStatement,
      AnalyzedObjectsAsFortran *asFortran)
      : out_{out}, langOpts_{langOpts}, indentationAmount_{indentationAmount},
        encoding_{encoding}, capitalizeKeywords_{capitalize},
        backslashEscapes_{backslashEscapes}, preStatement_{preStatement},
        asFortran_{asFortran} {}

  // In nearly all cases, this code avoids defining Boolean-valued Pre()
  // callbacks for the parse tree walking framework in favor of two void
  // functions, Before() and Unparse(), which imply true and false return
  // values for Pre() respectively.
  template <typename T> void Before(const T &) {}
  template <typename T> double Unparse(const T &); // not void, never used

  template <typename T> bool Pre(const T &x) {
    if constexpr (std::is_void_v<decltype(Unparse(x))>) {
      // There is a local definition of Unparse() for this type.  It
      // overrides the parse tree walker's default Walk() over the descendents.
      Before(x);
      Unparse(x);
      Post(x);
      return false; // Walk() does not visit descendents
    } else if constexpr (HasTypedExpr<T>::value) {
      // Format the expression representation from semantics
      if (asFortran_ && x.typedExpr) {
        asFortran_->expr(out_, *x.typedExpr);
        return false;
      } else {
        return true;
      }
    } else {
      Before(x);
      return true; // there's no Unparse() defined here, Walk() the descendents
    }
  }
  template <typename T> void Post(const T &) {}

  // Emit simple types as-is.
  void Unparse(const std::string &x) { Put(x); }
  void Unparse(int x) { Put(std::to_string(x)); }
  void Unparse(unsigned int x) { Put(std::to_string(x)); }
  void Unparse(long x) { Put(std::to_string(x)); }
  void Unparse(unsigned long x) { Put(std::to_string(x)); }
  void Unparse(long long x) { Put(std::to_string(x)); }
  void Unparse(unsigned long long x) { Put(std::to_string(x)); }
  void Unparse(char x) { Put(x); }

  // Statement labels and ends of lines
  template <typename T> void Before(const Statement<T> &x) {
    if (preStatement_) {
      (*preStatement_)(x.source, out_, indent_);
    }
    Walk(x.label, " ");
  }
  template <typename T> void Post(const Statement<T> &) { Put('\n'); }

  // The special-case formatting functions for these productions are
  // ordered to correspond roughly to their order of appearance in
  // the Fortran 2018 standard (and parse-tree.h).

  void Unparse(const Program &x) { // R501
    Walk("", x.v, "\n"); // put blank lines between ProgramUnits
  }

  void Unparse(const Name &x) { // R603
    Put(x.ToString());
  }
  void Unparse(const DefinedOperator::IntrinsicOperator &x) { // R608
    switch (x) {
    case DefinedOperator::IntrinsicOperator::Power:
      Put("**");
      break;
    case DefinedOperator::IntrinsicOperator::Multiply:
      Put('*');
      break;
    case DefinedOperator::IntrinsicOperator::Divide:
      Put('/');
      break;
    case DefinedOperator::IntrinsicOperator::Add:
      Put('+');
      break;
    case DefinedOperator::IntrinsicOperator::Subtract:
      Put('-');
      break;
    case DefinedOperator::IntrinsicOperator::Concat:
      Put("//");
      break;
    case DefinedOperator::IntrinsicOperator::LT:
      Put('<');
      break;
    case DefinedOperator::IntrinsicOperator::LE:
      Put("<=");
      break;
    case DefinedOperator::IntrinsicOperator::EQ:
      Put("==");
      break;
    case DefinedOperator::IntrinsicOperator::NE:
      Put("/=");
      break;
    case DefinedOperator::IntrinsicOperator::GE:
      Put(">=");
      break;
    case DefinedOperator::IntrinsicOperator::GT:
      Put('>');
      break;
    default:
      Put('.'), Word(DefinedOperator::EnumToString(x)), Put('.');
    }
  }
  void Post(const Star &) { Put('*'); } // R701 &c.
  void Post(const TypeParamValue::Deferred &) { Put(':'); } // R701
  void Unparse(const DeclarationTypeSpec::Type &x) { // R703
    Word("TYPE("), Walk(x.derived), Put(')');
  }
  void Unparse(const DeclarationTypeSpec::Class &x) {
    Word("CLASS("), Walk(x.derived), Put(')');
  }
  void Post(const DeclarationTypeSpec::ClassStar &) { Word("CLASS(*)"); }
  void Post(const DeclarationTypeSpec::TypeStar &) { Word("TYPE(*)"); }
  void Unparse(const DeclarationTypeSpec::Record &x) {
    Word("RECORD/"), Walk(x.v), Put('/');
  }
  void Before(const IntrinsicTypeSpec::Real &) { // R704
    Word("REAL");
  }
  void Before(const IntrinsicTypeSpec::Complex &) { Word("COMPLEX"); }
  void Post(const IntrinsicTypeSpec::DoublePrecision &) {
    Word("DOUBLE PRECISION");
  }
  void Before(const IntrinsicTypeSpec::Character &) { Word("CHARACTER"); }
  void Before(const IntrinsicTypeSpec::Logical &) { Word("LOGICAL"); }
  void Post(const IntrinsicTypeSpec::DoubleComplex &) {
    Word("DOUBLE COMPLEX");
  }
  void Before(const UnsignedTypeSpec &) { Word("UNSIGNED"); }
  void Before(const IntrinsicVectorTypeSpec &) { Word("VECTOR("); }
  void Post(const IntrinsicVectorTypeSpec &) { Put(')'); }
  void Post(const VectorTypeSpec::PairVectorTypeSpec &) {
    Word("__VECTOR_PAIR");
  }
  void Post(const VectorTypeSpec::QuadVectorTypeSpec &) {
    Word("__VECTOR_QUAD");
  }
  void Before(const IntegerTypeSpec &) { // R705
    Word("INTEGER");
  }
  void Unparse(const KindSelector &x) { // R706
    common::visit(
        common::visitors{
            [&](const ScalarIntConstantExpr &y) {
              Put('('), Word("KIND="), Walk(y), Put(')');
            },
            [&](const KindSelector::StarSize &y) { Put('*'), Walk(y.v); },
        },
        x.u);
  }
  void Unparse(const SignedIntLiteralConstant &x) { // R707
    Put(std::get<CharBlock>(x.t).ToString());
    Walk("_", std::get<std::optional<KindParam>>(x.t));
  }
  void Unparse(const IntLiteralConstant &x) { // R708
    Put(std::get<CharBlock>(x.t).ToString());
    Walk("_", std::get<std::optional<KindParam>>(x.t));
  }
  void Unparse(const Sign &x) { // R712
    Put(x == Sign::Negative ? '-' : '+');
  }
  void Unparse(const RealLiteralConstant &x) { // R714, R715
    Put(x.real.source.ToString()), Walk("_", x.kind);
  }
  void Unparse(const ComplexLiteralConstant &x) { // R718 - R720
    Put('('), Walk(x.t, ","), Put(')');
  }
  void Unparse(const CharSelector::LengthAndKind &x) { // R721
    Put('('), Word("KIND="), Walk(x.kind);
    Walk(", LEN=", x.length), Put(')');
  }
  void Unparse(const LengthSelector &x) { // R722
    common::visit(common::visitors{
                      [&](const TypeParamValue &y) {
                        Put('('), Word("LEN="), Walk(y), Put(')');
                      },
                      [&](const CharLength &y) { Put('*'), Walk(y); },
                  },
        x.u);
  }
  void Unparse(const CharLength &x) { // R723
    common::visit(
        common::visitors{
            [&](const TypeParamValue &y) { Put('('), Walk(y), Put(')'); },
            [&](const std::int64_t &y) { Walk(y); },
        },
        x.u);
  }
  void Unparse(const CharLiteralConstant &x) { // R724
    const auto &str{std::get<std::string>(x.t)};
    if (const auto &k{std::get<std::optional<KindParam>>(x.t)}) {
      Walk(*k), Put('_');
    }
    PutNormalized(str);
  }
  void Unparse(const HollerithLiteralConstant &x) {
    auto ucs{DecodeString<std::u32string, Encoding::UTF_8>(x.v, false)};
    Unparse(ucs.size());
    Put('H');
    for (char32_t ch : ucs) {
      EncodedCharacter encoded{EncodeCharacter(encoding_, ch)};
      for (int j{0}; j < encoded.bytes; ++j) {
        Put(encoded.buffer[j]);
      }
    }
  }
  void Unparse(const LogicalLiteralConstant &x) { // R725
    Put(std::get<bool>(x.t) ? ".TRUE." : ".FALSE.");
    Walk("_", std::get<std::optional<KindParam>>(x.t));
  }
  void Unparse(const DerivedTypeStmt &x) { // R727
    Word("TYPE"), Walk(", ", std::get<std::list<TypeAttrSpec>>(x.t), ", ");
    Put(" :: "), Walk(std::get<Name>(x.t));
    Walk("(", std::get<std::list<Name>>(x.t), ", ", ")");
    Indent();
  }
  void Unparse(const Abstract &) { // R728, &c.
    Word("ABSTRACT");
  }
  void Post(const TypeAttrSpec::BindC &) { Word("BIND(C)"); }
  void Unparse(const TypeAttrSpec::Extends &x) {
    Word("EXTENDS("), Walk(x.v), Put(')');
  }
  void Unparse(const EndTypeStmt &x) { // R730
    Outdent(), Word("END TYPE"), Walk(" ", x.v);
  }
  void Unparse(const SequenceStmt &) { // R731
    Word("SEQUENCE");
  }
  void Unparse(const TypeParamDefStmt &x) { // R732
    Walk(std::get<IntegerTypeSpec>(x.t));
    Put(", "), Walk(std::get<common::TypeParamAttr>(x.t));
    Put(" :: "), Walk(std::get<std::list<TypeParamDecl>>(x.t), ", ");
  }
  void Unparse(const TypeParamDecl &x) { // R733
    Walk(std::get<Name>(x.t));
    Walk("=", std::get<std::optional<ScalarIntConstantExpr>>(x.t));
  }
  void Unparse(const DataComponentDefStmt &x) { // R737
    const auto &dts{std::get<DeclarationTypeSpec>(x.t)};
    const auto &attrs{std::get<std::list<ComponentAttrSpec>>(x.t)};
    const auto &decls{std::get<std::list<ComponentOrFill>>(x.t)};
    Walk(dts), Walk(", ", attrs, ", ");
    if (!attrs.empty() ||
        (!std::holds_alternative<DeclarationTypeSpec::Record>(dts.u) &&
            std::none_of(
                decls.begin(), decls.end(), [](const ComponentOrFill &c) {
                  return common::visit(
                      common::visitors{
                          [](const ComponentDecl &d) {
                            const auto &init{
                                std::get<std::optional<Initialization>>(d.t)};
                            return init &&
                                std::holds_alternative<std::list<
                                    common::Indirection<DataStmtValue>>>(
                                    init->u);
                          },
                          [](const FillDecl &) { return false; },
                      },
                      c.u);
                }))) {
      Put(" ::");
    }
    Put(' '), Walk(decls, ", ");
  }
  void Unparse(const Allocatable &) { // R738
    Word("ALLOCATABLE");
  }
  void Unparse(const Pointer &) { Word("POINTER"); }
  void Unparse(const Contiguous &) { Word("CONTIGUOUS"); }
  void Before(const ComponentAttrSpec &x) {
    common::visit(common::visitors{
                      [&](const CoarraySpec &) { Word("CODIMENSION["); },
                      [&](const ComponentArraySpec &) { Word("DIMENSION("); },
                      [](const auto &) {},
                  },
        x.u);
  }
  void Post(const ComponentAttrSpec &x) {
    common::visit(common::visitors{
                      [&](const CoarraySpec &) { Put(']'); },
                      [&](const ComponentArraySpec &) { Put(')'); },
                      [](const auto &) {},
                  },
        x.u);
  }
  void Unparse(const ComponentDecl &x) { // R739
    Walk(std::get<ObjectName>(x.t));
    Walk("(", std::get<std::optional<ComponentArraySpec>>(x.t), ")");
    Walk("[", std::get<std::optional<CoarraySpec>>(x.t), "]");
    Walk("*", std::get<std::optional<CharLength>>(x.t));
    Walk(std::get<std::optional<Initialization>>(x.t));
  }
  void Unparse(const FillDecl &x) { // DEC extension
    Put("%FILL");
    Walk("(", std::get<std::optional<ComponentArraySpec>>(x.t), ")");
    Walk("*", std::get<std::optional<CharLength>>(x.t));
  }
  void Unparse(const ComponentArraySpec &x) { // R740
    common::visit(
        common::visitors{
            [&](const std::list<ExplicitShapeSpec> &y) { Walk(y, ","); },
            [&](const DeferredShapeSpecList &y) { Walk(y); },
        },
        x.u);
  }
  void Unparse(const ProcComponentDefStmt &x) { // R741
    Word("PROCEDURE(");
    Walk(std::get<std::optional<ProcInterface>>(x.t)), Put(')');
    Walk(", ", std::get<std::list<ProcComponentAttrSpec>>(x.t), ", ");
    Put(" :: "), Walk(std::get<std::list<ProcDecl>>(x.t), ", ");
  }
  void Unparse(const NoPass &) { // R742
    Word("NOPASS");
  }
  void Unparse(const Pass &x) { Word("PASS"), Walk("(", x.v, ")"); }
  void Unparse(const Initialization &x) { // R743 & R805
    common::visit(
        common::visitors{
            [&](const ConstantExpr &y) { Put(" = "), Walk(y); },
            [&](const NullInit &y) { Put(" => "), Walk(y); },
            [&](const InitialDataTarget &y) { Put(" => "), Walk(y); },
            [&](const std::list<common::Indirection<DataStmtValue>> &y) {
              Walk("/", y, ", ", "/");
            },
        },
        x.u);
  }
  void Unparse(const PrivateStmt &) { // R745
    Word("PRIVATE");
  }
  void Unparse(const TypeBoundProcedureStmt::WithoutInterface &x) { // R749
    Word("PROCEDURE"), Walk(", ", x.attributes, ", ");
    Put(" :: "), Walk(x.declarations, ", ");
  }
  void Unparse(const TypeBoundProcedureStmt::WithInterface &x) {
    Word("PROCEDURE("), Walk(x.interfaceName), Put("), ");
    Walk(x.attributes);
    Put(" :: "), Walk(x.bindingNames, ", ");
  }
  void Unparse(const TypeBoundProcDecl &x) { // R750
    Walk(std::get<Name>(x.t));
    Walk(" => ", std::get<std::optional<Name>>(x.t));
  }
  void Unparse(const TypeBoundGenericStmt &x) { // R751
    Word("GENERIC"), Walk(", ", std::get<std::optional<AccessSpec>>(x.t));
    Put(" :: "), Walk(std::get<common::Indirection<GenericSpec>>(x.t));
    Put(" => "), Walk(std::get<std::list<Name>>(x.t), ", ");
  }
  void Post(const BindAttr::Deferred &) { Word("DEFERRED"); } // R752
  void Post(const BindAttr::Non_Overridable &) { Word("NON_OVERRIDABLE"); }
  void Unparse(const FinalProcedureStmt &x) { // R753
    Word("FINAL :: "), Walk(x.v, ", ");
  }
  void Unparse(const DerivedTypeSpec &x) { // R754
    Walk(std::get<Name>(x.t));
    Walk("(", std::get<std::list<TypeParamSpec>>(x.t), ",", ")");
  }
  void Unparse(const TypeParamSpec &x) { // R755
    Walk(std::get<std::optional<Keyword>>(x.t), "=");
    Walk(std::get<TypeParamValue>(x.t));
  }
  void Unparse(const StructureConstructor &x) { // R756
    Walk(std::get<DerivedTypeSpec>(x.t));
    Put('('), Walk(std::get<std::list<ComponentSpec>>(x.t), ", "), Put(')');
  }
  void Unparse(const ComponentSpec &x) { // R757
    Walk(std::get<std::optional<Keyword>>(x.t), "=");
    Walk(std::get<ComponentDataSource>(x.t));
  }
  void Unparse(const EnumDefStmt &) { // R760
    Word("ENUM, BIND(C)"), Indent();
  }
  void Unparse(const EnumeratorDefStmt &x) { // R761
    Word("ENUMERATOR :: "), Walk(x.v, ", ");
  }
  void Unparse(const Enumerator &x) { // R762
    Walk(std::get<NamedConstant>(x.t));
    Walk(" = ", std::get<std::optional<ScalarIntConstantExpr>>(x.t));
  }
  void Post(const EndEnumStmt &) { // R763
    Outdent(), Word("END ENUM");
  }
  void Unparse(const BOZLiteralConstant &x) { // R764 - R767
    Put(x.v);
  }
  void Unparse(const AcValue::Triplet &x) { // R773
    Walk(std::get<0>(x.t)), Put(':'), Walk(std::get<1>(x.t));
    Walk(":", std::get<std::optional<ScalarIntExpr>>(x.t));
  }
  void Unparse(const ArrayConstructor &x) { // R769
    Put('['), Walk(x.v), Put(']');
  }
  void Unparse(const AcSpec &x) { // R770
    Walk(x.type, "::"), Walk(x.values, ", ");
  }
  template <typename A, typename B> void Unparse(const LoopBounds<A, B> &x) {
    Walk(x.name), Put('='), Walk(x.lower), Put(','), Walk(x.upper);
    Walk(",", x.step);
  }
  void Unparse(const AcImpliedDo &x) { // R774
    Put('('), Walk(std::get<std::list<AcValue>>(x.t), ", ");
    Put(", "), Walk(std::get<AcImpliedDoControl>(x.t)), Put(')');
  }
  void Unparse(const AcImpliedDoControl &x) { // R775
    Walk(std::get<std::optional<IntegerTypeSpec>>(x.t), "::");
    Walk(std::get<AcImpliedDoControl::Bounds>(x.t));
  }

  void Unparse(const TypeDeclarationStmt &x) { // R801
    const auto &dts{std::get<DeclarationTypeSpec>(x.t)};
    const auto &attrs{std::get<std::list<AttrSpec>>(x.t)};
    const auto &decls{std::get<std::list<EntityDecl>>(x.t)};
    Walk(dts), Walk(", ", attrs, ", ");

    static const auto isInitializerOldStyle{[](const Initialization &i) {
      return std::holds_alternative<
          std::list<common::Indirection<DataStmtValue>>>(i.u);
    }};
    static const auto hasAssignmentInitializer{[](const EntityDecl &d) {
      // Does a declaration have a new-style =x initializer?
      const auto &init{std::get<std::optional<Initialization>>(d.t)};
      return init && !isInitializerOldStyle(*init);
    }};
    static const auto hasSlashDelimitedInitializer{[](const EntityDecl &d) {
      // Does a declaration have an old-style /x/ initializer?
      const auto &init{std::get<std::optional<Initialization>>(d.t)};
      return init && isInitializerOldStyle(*init);
    }};
    const auto useDoubledColons{[&]() {
      bool isRecord{std::holds_alternative<DeclarationTypeSpec::Record>(dts.u)};
      if (!attrs.empty()) {
        // Attributes after the type require :: before the entities.
        CHECK(!isRecord);
        return true;
      }
      if (std::any_of(decls.begin(), decls.end(), hasAssignmentInitializer)) {
        // Always use :: with new style standard initializers (=x),
        // since the standard requires them to appear (even in free form,
        // where mandatory spaces already disambiguate INTEGER J=666).
        CHECK(!isRecord);
        return true;
      }
      if (isRecord) {
        // Never put :: in a legacy extension RECORD// statement.
        return false;
      }
      // The :: is optional for this declaration.  Avoid usage that can
      // crash the pgf90 compiler.
      if (std::any_of(
              decls.begin(), decls.end(), hasSlashDelimitedInitializer)) {
        // Don't use :: when a declaration uses legacy DATA-statement-like
        // /x/ initialization.
        return false;
      }
      // Don't use :: with intrinsic types.  Otherwise, use it.
      return !std::holds_alternative<IntrinsicTypeSpec>(dts.u);
    }};

    if (useDoubledColons()) {
      Put(" ::");
    }
    Put(' '), Walk(std::get<std::list<EntityDecl>>(x.t), ", ");
  }
  void Before(const AttrSpec &x) { // R802
    common::visit(common::visitors{
                      [&](const CoarraySpec &) { Word("CODIMENSION["); },
                      [&](const ArraySpec &) { Word("DIMENSION("); },
                      [](const auto &) {},
                  },
        x.u);
  }
  void Post(const AttrSpec &x) {
    common::visit(common::visitors{
                      [&](const CoarraySpec &) { Put(']'); },
                      [&](const ArraySpec &) { Put(')'); },
                      [](const auto &) {},
                  },
        x.u);
  }
  void Unparse(const EntityDecl &x) { // R803
    Walk(std::get<ObjectName>(x.t));
    Walk("(", std::get<std::optional<ArraySpec>>(x.t), ")");
    Walk("[", std::get<std::optional<CoarraySpec>>(x.t), "]");
    Walk("*", std::get<std::optional<CharLength>>(x.t));
    Walk(std::get<std::optional<Initialization>>(x.t));
  }
  void Unparse(const NullInit &) { // R806
    Word("NULL()");
  }
  void Unparse(const LanguageBindingSpec &x) { // R808 & R1528
    Word("BIND(C");
    Walk(
        ", NAME=", std::get<std::optional<ScalarDefaultCharConstantExpr>>(x.t));
    if (std::get<bool>(x.t)) {
      Word(", CDEFINED");
    }
    Put(')');
  }
  void Unparse(const CoarraySpec &x) { // R809
    common::visit(common::visitors{
                      [&](const DeferredCoshapeSpecList &y) { Walk(y); },
                      [&](const ExplicitCoshapeSpec &y) { Walk(y); },
                  },
        x.u);
  }
  void Unparse(const DeferredCoshapeSpecList &x) { // R810
    for (auto j{x.v}; j > 0; --j) {
      Put(':');
      if (j > 1) {
        Put(',');
      }
    }
  }
  void Unparse(const ExplicitCoshapeSpec &x) { // R811
    Walk(std::get<std::list<ExplicitShapeSpec>>(x.t), ",", ",");
    Walk(std::get<std::optional<SpecificationExpr>>(x.t), ":"), Put('*');
  }
  void Unparse(const ExplicitShapeSpec &x) { // R812 - R813 & R816 - R818
    Walk(std::get<std::optional<SpecificationExpr>>(x.t), ":");
    Walk(std::get<SpecificationExpr>(x.t));
  }
  void Unparse(const ArraySpec &x) { // R815
    common::visit(
        common::visitors{
            [&](const std::list<ExplicitShapeSpec> &y) { Walk(y, ","); },
            [&](const std::list<AssumedShapeSpec> &y) { Walk(y, ","); },
            [&](const DeferredShapeSpecList &y) { Walk(y); },
            [&](const AssumedSizeSpec &y) { Walk(y); },
            [&](const ImpliedShapeSpec &y) { Walk(y); },
            [&](const AssumedRankSpec &y) { Walk(y); },
        },
        x.u);
  }
  void Post(const AssumedShapeSpec &) { Put(':'); } // R819
  void Unparse(const DeferredShapeSpecList &x) { // R820
    for (auto j{x.v}; j > 0; --j) {
      Put(':');
      if (j > 1) {
        Put(',');
      }
    }
  }
  void Unparse(const AssumedImpliedSpec &x) { // R821
    Walk(x.v, ":");
    Put('*');
  }
  void Unparse(const AssumedSizeSpec &x) { // R822
    Walk(std::get<std::list<ExplicitShapeSpec>>(x.t), ",", ",");
    Walk(std::get<AssumedImpliedSpec>(x.t));
  }
  void Unparse(const ImpliedShapeSpec &x) { // R823
    Walk(x.v, ",");
  }
  void Post(const AssumedRankSpec &) { Put(".."); } // R825
  void Post(const Asynchronous &) { Word("ASYNCHRONOUS"); }
  void Post(const External &) { Word("EXTERNAL"); }
  void Post(const Intrinsic &) { Word("INTRINSIC"); }
  void Post(const Optional &) { Word("OPTIONAL"); }
  void Post(const Parameter &) { Word("PARAMETER"); }
  void Post(const Protected &) { Word("PROTECTED"); }
  void Post(const Save &) { Word("SAVE"); }
  void Post(const Target &) { Word("TARGET"); }
  void Post(const Value &) { Word("VALUE"); }
  void Post(const Volatile &) { Word("VOLATILE"); }
  void Unparse(const IntentSpec &x) { // R826
    Word("INTENT("), Walk(x.v), Put(")");
  }
  void Unparse(const AccessStmt &x) { // R827
    Walk(std::get<AccessSpec>(x.t));
    Walk(" :: ", std::get<std::list<AccessId>>(x.t), ", ");
  }
  void Unparse(const AllocatableStmt &x) { // R829
    Word("ALLOCATABLE :: "), Walk(x.v, ", ");
  }
  void Unparse(const ObjectDecl &x) { // R830 & R860
    Walk(std::get<ObjectName>(x.t));
    Walk("(", std::get<std::optional<ArraySpec>>(x.t), ")");
    Walk("[", std::get<std::optional<CoarraySpec>>(x.t), "]");
  }
  void Unparse(const AsynchronousStmt &x) { // R831
    Word("ASYNCHRONOUS :: "), Walk(x.v, ", ");
  }
  void Unparse(const BindStmt &x) { // R832
    Walk(x.t, " :: ");
  }
  void Unparse(const BindEntity &x) { // R833
    bool isCommon{std::get<BindEntity::Kind>(x.t) == BindEntity::Kind::Common};
    const char *slash{isCommon ? "/" : ""};
    Put(slash), Walk(std::get<Name>(x.t)), Put(slash);
  }
  void Unparse(const CodimensionStmt &x) { // R834
    Word("CODIMENSION :: "), Walk(x.v, ", ");
  }
  void Unparse(const CodimensionDecl &x) { // R835
    Walk(std::get<Name>(x.t));
    Put('['), Walk(std::get<CoarraySpec>(x.t)), Put(']');
  }
  void Unparse(const ContiguousStmt &x) { // R836
    Word("CONTIGUOUS :: "), Walk(x.v, ", ");
  }
  void Unparse(const DataStmt &x) { // R837
    Word("DATA "), Walk(x.v, ", ");
  }
  void Unparse(const DataStmtSet &x) { // R838
    Walk(std::get<std::list<DataStmtObject>>(x.t), ", ");
    Put('/'), Walk(std::get<std::list<DataStmtValue>>(x.t), ", "), Put('/');
  }
  void Unparse(const DataImpliedDo &x) { // R840, R842
    Put('('), Walk(std::get<std::list<DataIDoObject>>(x.t), ", "), Put(',');
    Walk(std::get<std::optional<IntegerTypeSpec>>(x.t), "::");
    Walk(std::get<DataImpliedDo::Bounds>(x.t)), Put(')');
  }
  void Unparse(const DataStmtValue &x) { // R843
    Walk(std::get<std::optional<DataStmtRepeat>>(x.t), "*");
    Walk(std::get<DataStmtConstant>(x.t));
  }
  void Unparse(const DimensionStmt &x) { // R848
    Word("DIMENSION :: "), Walk(x.v, ", ");
  }
  void Unparse(const DimensionStmt::Declaration &x) {
    Walk(std::get<Name>(x.t));
    Put('('), Walk(std::get<ArraySpec>(x.t)), Put(')');
  }
  void Unparse(const IntentStmt &x) { // R849
    Walk(x.t, " :: ");
  }
  void Unparse(const OptionalStmt &x) { // R850
    Word("OPTIONAL :: "), Walk(x.v, ", ");
  }
  void Unparse(const ParameterStmt &x) { // R851
    Word("PARAMETER("), Walk(x.v, ", "), Put(')');
  }
  void Unparse(const NamedConstantDef &x) { // R852
    Walk(x.t, "=");
  }
  void Unparse(const PointerStmt &x) { // R853
    Word("POINTER :: "), Walk(x.v, ", ");
  }
  void Unparse(const PointerDecl &x) { // R854
    Walk(std::get<Name>(x.t));
    Walk("(", std::get<std::optional<DeferredShapeSpecList>>(x.t), ")");
  }
  void Unparse(const ProtectedStmt &x) { // R855
    Word("PROTECTED :: "), Walk(x.v, ", ");
  }
  void Unparse(const SaveStmt &x) { // R856
    Word("SAVE"), Walk(" :: ", x.v, ", ");
  }
  void Unparse(const SavedEntity &x) { // R857, R858
    bool isCommon{
        std::get<SavedEntity::Kind>(x.t) == SavedEntity::Kind::Common};
    const char *slash{isCommon ? "/" : ""};
    Put(slash), Walk(std::get<Name>(x.t)), Put(slash);
  }
  void Unparse(const TargetStmt &x) { // R859
    Word("TARGET :: "), Walk(x.v, ", ");
  }
  void Unparse(const ValueStmt &x) { // R861
    Word("VALUE :: "), Walk(x.v, ", ");
  }
  void Unparse(const VolatileStmt &x) { // R862
    Word("VOLATILE :: "), Walk(x.v, ", ");
  }
  void Unparse(const ImplicitStmt &x) { // R863
    Word("IMPLICIT ");
    common::visit(
        common::visitors{
            [&](const std::list<ImplicitSpec> &y) { Walk(y, ", "); },
            [&](const std::list<ImplicitStmt::ImplicitNoneNameSpec> &y) {
              Word("NONE"), Walk(" (", y, ", ", ")");
            },
        },
        x.u);
  }
  void Unparse(const ImplicitSpec &x) { // R864
    Walk(std::get<DeclarationTypeSpec>(x.t));
    Put('('), Walk(std::get<std::list<LetterSpec>>(x.t), ", "), Put(')');
  }
  void Unparse(const LetterSpec &x) { // R865
    Put(*std::get<const char *>(x.t));
    auto second{std::get<std::optional<const char *>>(x.t)};
    if (second) {
      Put('-'), Put(**second);
    }
  }
  void Unparse(const ImportStmt &x) { // R867
    Word("IMPORT");
    switch (x.kind) {
    case common::ImportKind::Default:
      Walk(" :: ", x.names, ", ");
      break;
    case common::ImportKind::Only:
      Put(", "), Word("ONLY: ");
      Walk(x.names, ", ");
      break;
    case common::ImportKind::None:
      Word(", NONE");
      break;
    case common::ImportKind::All:
      Word(", ALL");
      break;
    }
  }
  void Unparse(const NamelistStmt &x) { // R868
    Word("NAMELIST"), Walk(x.v, ", ");
  }
  void Unparse(const NamelistStmt::Group &x) {
    Put('/'), Walk(std::get<Name>(x.t)), Put('/');
    Walk(std::get<std::list<Name>>(x.t), ", ");
  }
  void Unparse(const EquivalenceStmt &x) { // R870, R871
    Word("EQUIVALENCE");
    const char *separator{" "};
    for (const std::list<EquivalenceObject> &y : x.v) {
      Put(separator), Put('('), Walk(y), Put(')');
      separator = ", ";
    }
  }
  void Unparse(const CommonStmt &x) { // R873
    Word("COMMON ");
    Walk(x.blocks);
  }
  void Unparse(const CommonBlockObject &x) { // R874
    Walk(std::get<Name>(x.t));
    Walk("(", std::get<std::optional<ArraySpec>>(x.t), ")");
  }
  void Unparse(const CommonStmt::Block &x) {
    Word("/"), Walk(std::get<std::optional<Name>>(x.t)), Word("/");
    Walk(std::get<std::list<CommonBlockObject>>(x.t));
  }

  void Unparse(const Substring &x) { // R908, R909
    Walk(std::get<DataRef>(x.t));
    Put('('), Walk(std::get<SubstringRange>(x.t)), Put(')');
  }
  void Unparse(const CharLiteralConstantSubstring &x) {
    Walk(std::get<CharLiteralConstant>(x.t));
    Put('('), Walk(std::get<SubstringRange>(x.t)), Put(')');
  }
  void Unparse(const SubstringInquiry &x) {
    Walk(x.v);
    Put(x.source.back() == 'n' ? "%LEN" : "%KIND");
  }
  void Unparse(const SubstringRange &x) { // R910
    Walk(x.t, ":");
  }
  void Unparse(const PartRef &x) { // R912
    Walk(x.name);
    Walk("(", x.subscripts, ",", ")");
    Walk(x.imageSelector);
  }
  void Unparse(const StructureComponent &x) { // R913
    Walk(x.base);
    if (structureComponents_.find(x.component.source) !=
        structureComponents_.end()) {
      Put('.');
    } else {
      Put('%');
    }
    Walk(x.component);
  }
  void Unparse(const ArrayElement &x) { // R917
    Walk(x.base);
    Put('('), Walk(x.subscripts, ","), Put(')');
  }
  void Unparse(const SubscriptTriplet &x) { // R921
    Walk(std::get<0>(x.t)), Put(':'), Walk(std::get<1>(x.t));
    Walk(":", std::get<2>(x.t));
  }
  void Unparse(const ImageSelector &x) { // R924
    Put('['), Walk(std::get<std::list<Cosubscript>>(x.t), ",");
    Walk(",", std::get<std::list<ImageSelectorSpec>>(x.t), ","), Put(']');
  }
  void Before(const ImageSelectorSpec::Stat &) { // R926
    Word("STAT=");
  }
  void Before(const ImageSelectorSpec::Team_Number &) { Word("TEAM_NUMBER="); }
  void Before(const ImageSelectorSpec &x) {
    if (std::holds_alternative<TeamValue>(x.u)) {
      Word("TEAM=");
    }
  }
  void Unparse(const AllocateStmt &x) { // R927
    Word("ALLOCATE(");
    Walk(std::get<std::optional<TypeSpec>>(x.t), "::");
    Walk(std::get<std::list<Allocation>>(x.t), ", ");
    Walk(", ", std::get<std::list<AllocOpt>>(x.t), ", "), Put(')');
  }
  void Before(const AllocOpt &x) { // R928, R931
    common::visit(common::visitors{
                      [&](const AllocOpt::Mold &) { Word("MOLD="); },
                      [&](const AllocOpt::Source &) { Word("SOURCE="); },
                      [&](const AllocOpt::Stream &) { Word("STREAM="); },
                      [&](const AllocOpt::Pinned &) { Word("PINNED="); },
                      [](const StatOrErrmsg &) {},
                  },
        x.u);
  }
  void Unparse(const Allocation &x) { // R932
    Walk(std::get<AllocateObject>(x.t));
    Walk("(", std::get<std::list<AllocateShapeSpec>>(x.t), ",", ")");
    Walk("[", std::get<std::optional<AllocateCoarraySpec>>(x.t), "]");
  }
  void Unparse(const AllocateShapeSpec &x) { // R934 & R938
    Walk(std::get<std::optional<BoundExpr>>(x.t), ":");
    Walk(std::get<BoundExpr>(x.t));
  }
  void Unparse(const AllocateCoarraySpec &x) { // R937
    Walk(std::get<std::list<AllocateCoshapeSpec>>(x.t), ",", ",");
    Walk(std::get<std::optional<BoundExpr>>(x.t), ":"), Put('*');
  }
  void Unparse(const NullifyStmt &x) { // R939
    Word("NULLIFY("), Walk(x.v, ", "), Put(')');
  }
  void Unparse(const DeallocateStmt &x) { // R941
    Word("DEALLOCATE(");
    Walk(std::get<std::list<AllocateObject>>(x.t), ", ");
    Walk(", ", std::get<std::list<StatOrErrmsg>>(x.t), ", "), Put(')');
  }
  void Before(const StatOrErrmsg &x) { // R942 & R1165
    common::visit(common::visitors{
                      [&](const StatVariable &) { Word("STAT="); },
                      [&](const MsgVariable &) { Word("ERRMSG="); },
                  },
        x.u);
  }

  // R1001 - R1022
  void Unparse(const Expr::Parentheses &x) { Put('('), Walk(x.v), Put(')'); }
  void Before(const Expr::UnaryPlus &) { Put("+"); }
  void Before(const Expr::Negate &) { Put("-"); }
  void Before(const Expr::NOT &) { Word(".NOT."); }
  void Unparse(const Expr::PercentLoc &x) {
    Word("%LOC("), Walk(x.v), Put(')');
  }
  void Unparse(const Expr::Power &x) { Walk(x.t, "**"); }
  void Unparse(const Expr::Multiply &x) { Walk(x.t, "*"); }
  void Unparse(const Expr::Divide &x) { Walk(x.t, "/"); }
  void Unparse(const Expr::Add &x) { Walk(x.t, "+"); }
  void Unparse(const Expr::Subtract &x) { Walk(x.t, "-"); }
  void Unparse(const Expr::Concat &x) { Walk(x.t, "//"); }
  void Unparse(const Expr::LT &x) { Walk(x.t, "<"); }
  void Unparse(const Expr::LE &x) { Walk(x.t, "<="); }
  void Unparse(const Expr::EQ &x) { Walk(x.t, "=="); }
  void Unparse(const Expr::NE &x) { Walk(x.t, "/="); }
  void Unparse(const Expr::GE &x) { Walk(x.t, ">="); }
  void Unparse(const Expr::GT &x) { Walk(x.t, ">"); }
  void Unparse(const Expr::AND &x) { Walk(x.t, ".AND."); }
  void Unparse(const Expr::OR &x) { Walk(x.t, ".OR."); }
  void Unparse(const Expr::EQV &x) { Walk(x.t, ".EQV."); }
  void Unparse(const Expr::NEQV &x) { Walk(x.t, ".NEQV."); }
  void Unparse(const Expr::ComplexConstructor &x) {
    Put('('), Walk(x.t, ","), Put(')');
  }
  void Unparse(const Expr::DefinedBinary &x) {
    Walk(std::get<1>(x.t)); // left
    Walk(std::get<DefinedOpName>(x.t));
    Walk(std::get<2>(x.t)); // right
  }
  void Unparse(const DefinedOpName &x) { // R1003, R1023, R1414, & R1415
    Walk(x.v);
  }
  void Unparse(const AssignmentStmt &x) { // R1032
    if (asFortran_ && x.typedAssignment.get()) {
      Put(' ');
      asFortran_->assignment(out_, *x.typedAssignment);
      Put('\n');
    } else {
      Walk(x.t, " = ");
    }
  }
  void Unparse(const PointerAssignmentStmt &x) { // R1033, R1034, R1038
    if (asFortran_ && x.typedAssignment.get()) {
      Put(' ');
      asFortran_->assignment(out_, *x.typedAssignment);
      Put('\n');
    } else {
      Walk(std::get<DataRef>(x.t));
      common::visit(
          common::visitors{
              [&](const std::list<BoundsRemapping> &y) {
                Put('('), Walk(y), Put(')');
              },
              [&](const std::list<BoundsSpec> &y) { Walk("(", y, ", ", ")"); },
          },
          std::get<PointerAssignmentStmt::Bounds>(x.t).u);
      Put(" => "), Walk(std::get<Expr>(x.t));
    }
  }
  void Post(const BoundsSpec &) { // R1035
    Put(':');
  }
  void Unparse(const BoundsRemapping &x) { // R1036
    Walk(x.t, ":");
  }
  void Unparse(const WhereStmt &x) { // R1041, R1045, R1046
    Word("WHERE ("), Walk(x.t, ") ");
  }
  void Unparse(const WhereConstructStmt &x) { // R1043
    Walk(std::get<std::optional<Name>>(x.t), ": ");
    Word("WHERE ("), Walk(std::get<LogicalExpr>(x.t)), Put(')');
    Indent();
  }
  void Unparse(const MaskedElsewhereStmt &x) { // R1047
    Outdent();
    Word("ELSEWHERE ("), Walk(std::get<LogicalExpr>(x.t)), Put(')');
    Walk(" ", std::get<std::optional<Name>>(x.t));
    Indent();
  }
  void Unparse(const ElsewhereStmt &x) { // R1048
    Outdent(), Word("ELSEWHERE"), Walk(" ", x.v), Indent();
  }
  void Unparse(const EndWhereStmt &x) { // R1049
    Outdent(), Word("END WHERE"), Walk(" ", x.v);
  }
  void Unparse(const ForallConstructStmt &x) { // R1051
    Walk(std::get<std::optional<Name>>(x.t), ": ");
    Word("FORALL"), Walk(std::get<common::Indirection<ConcurrentHeader>>(x.t));
    Indent();
  }
  void Unparse(const EndForallStmt &x) { // R1054
    Outdent(), Word("END FORALL"), Walk(" ", x.v);
  }
  void Before(const ForallStmt &) { // R1055
    Word("FORALL");
  }

  void Unparse(const AssociateStmt &x) { // R1103
    Walk(std::get<std::optional<Name>>(x.t), ": ");
    Word("ASSOCIATE (");
    Walk(std::get<std::list<Association>>(x.t), ", "), Put(')'), Indent();
  }
  void Unparse(const Association &x) { // R1104
    Walk(x.t, " => ");
  }
  void Unparse(const EndAssociateStmt &x) { // R1106
    Outdent(), Word("END ASSOCIATE"), Walk(" ", x.v);
  }
  void Unparse(const BlockStmt &x) { // R1108
    Walk(x.v, ": "), Word("BLOCK"), Indent();
  }
  void Unparse(const EndBlockStmt &x) { // R1110
    Outdent(), Word("END BLOCK"), Walk(" ", x.v);
  }
  void Unparse(const ChangeTeamStmt &x) { // R1112
    Walk(std::get<std::optional<Name>>(x.t), ": ");
    Word("CHANGE TEAM ("), Walk(std::get<TeamValue>(x.t));
    Walk(", ", std::get<std::list<CoarrayAssociation>>(x.t), ", ");
    Walk(", ", std::get<std::list<StatOrErrmsg>>(x.t), ", "), Put(')');
    Indent();
  }
  void Unparse(const CoarrayAssociation &x) { // R1113
    Walk(x.t, " => ");
  }
  void Unparse(const EndChangeTeamStmt &x) { // R1114
    Outdent(), Word("END TEAM (");
    Walk(std::get<std::list<StatOrErrmsg>>(x.t), ", ");
    Put(')'), Walk(" ", std::get<std::optional<Name>>(x.t));
  }
  void Unparse(const CriticalStmt &x) { // R1117
    Walk(std::get<std::optional<Name>>(x.t), ": ");
    Word("CRITICAL ("), Walk(std::get<std::list<StatOrErrmsg>>(x.t), ", ");
    Put(')'), Indent();
  }
  void Unparse(const EndCriticalStmt &x) { // R1118
    Outdent(), Word("END CRITICAL"), Walk(" ", x.v);
  }
  void Unparse(const DoConstruct &x) { // R1119, R1120
    Walk(std::get<Statement<NonLabelDoStmt>>(x.t));
    Indent(), Walk(std::get<Block>(x.t), ""), Outdent();
    Walk(std::get<Statement<EndDoStmt>>(x.t));
  }
  void Unparse(const LabelDoStmt &x) { // R1121
    Word("DO "), Walk(std::get<Label>(x.t));
    Walk(" ", std::get<std::optional<LoopControl>>(x.t));
  }
  void Unparse(const NonLabelDoStmt &x) { // R1122
    Walk(std::get<std::optional<Name>>(x.t), ": ");
    Word("DO ");
    Walk(std::get<std::optional<Label>>(x.t), " ");
    Walk(std::get<std::optional<LoopControl>>(x.t));
  }
  void Unparse(const LoopControl &x) { // R1123
    common::visit(common::visitors{
                      [&](const ScalarLogicalExpr &y) {
                        Word("WHILE ("), Walk(y), Put(')');
                      },
                      [&](const auto &y) { Walk(y); },
                  },
        x.u);
  }
  void Unparse(const ConcurrentHeader &x) { // R1125
    Put('('), Walk(std::get<std::optional<IntegerTypeSpec>>(x.t), "::");
    Walk(std::get<std::list<ConcurrentControl>>(x.t), ", ");
    Walk(", ", std::get<std::optional<ScalarLogicalExpr>>(x.t)), Put(')');
  }
  void Unparse(const ConcurrentControl &x) { // R1126 - R1128
    Walk(std::get<Name>(x.t)), Put('='), Walk(std::get<1>(x.t));
    Put(':'), Walk(std::get<2>(x.t));
    Walk(":", std::get<std::optional<ScalarIntExpr>>(x.t));
  }
  void Before(const LoopControl::Concurrent &) { // R1129
    Word("CONCURRENT");
  }
  void Unparse(const LocalitySpec::Local &x) {
    Word("LOCAL("), Walk(x.v, ", "), Put(')');
  }
  void Unparse(const LocalitySpec::LocalInit &x) {
    Word("LOCAL_INIT("), Walk(x.v, ", "), Put(')');
  }
  void Unparse(const LocalitySpec::Reduce &x) {
    Word("REDUCE("), Walk(std::get<parser::ReductionOperator>(x.t));
    Walk(":", std::get<std::list<parser::Name>>(x.t), ",", ")");
  }
  void Unparse(const LocalitySpec::Shared &x) {
    Word("SHARED("), Walk(x.v, ", "), Put(')');
  }
  void Post(const LocalitySpec::DefaultNone &) { Word("DEFAULT(NONE)"); }
  void Unparse(const EndDoStmt &x) { // R1132
    Word("END DO"), Walk(" ", x.v);
  }
  void Unparse(const CycleStmt &x) { // R1133
    Word("CYCLE"), Walk(" ", x.v);
  }
  void Unparse(const IfThenStmt &x) { // R1135
    Walk(std::get<std::optional<Name>>(x.t), ": ");
    Word("IF ("), Walk(std::get<ScalarLogicalExpr>(x.t));
    Put(") "), Word("THEN"), Indent();
  }
  void Unparse(const ElseIfStmt &x) { // R1136
    Outdent(), Word("ELSE IF (");
    Walk(std::get<ScalarLogicalExpr>(x.t)), Put(") "), Word("THEN");
    Walk(" ", std::get<std::optional<Name>>(x.t)), Indent();
  }
  void Unparse(const ElseStmt &x) { // R1137
    Outdent(), Word("ELSE"), Walk(" ", x.v), Indent();
  }
  void Unparse(const EndIfStmt &x) { // R1138
    Outdent(), Word("END IF"), Walk(" ", x.v);
  }
  void Unparse(const IfStmt &x) { // R1139
    Word("IF ("), Walk(x.t, ") ");
  }
  void Unparse(const SelectCaseStmt &x) { // R1141, R1144
    Walk(std::get<std::optional<Name>>(x.t), ": ");
    Word("SELECT CASE (");
    Walk(std::get<Scalar<Expr>>(x.t)), Put(')'), Indent();
  }
  void Unparse(const CaseStmt &x) { // R1142
    Outdent(), Word("CASE "), Walk(std::get<CaseSelector>(x.t));
    Walk(" ", std::get<std::optional<Name>>(x.t)), Indent();
  }
  void Unparse(const EndSelectStmt &x) { // R1143 & R1151 & R1155
    Outdent(), Word("END SELECT"), Walk(" ", x.v);
  }
  void Unparse(const CaseSelector &x) { // R1145
    common::visit(common::visitors{
                      [&](const std::list<CaseValueRange> &y) {
                        Put('('), Walk(y), Put(')');
                      },
                      [&](const Default &) { Word("DEFAULT"); },
                  },
        x.u);
  }
  void Unparse(const CaseValueRange::Range &x) { // R1146
    Walk(x.lower), Put(':'), Walk(x.upper);
  }
  void Unparse(const SelectRankStmt &x) { // R1149
    Walk(std::get<0>(x.t), ": ");
    Word("SELECT RANK ("), Walk(std::get<1>(x.t), " => ");
    Walk(std::get<Selector>(x.t)), Put(')'), Indent();
  }
  void Unparse(const SelectRankCaseStmt &x) { // R1150
    Outdent(), Word("RANK ");
    common::visit(common::visitors{
                      [&](const ScalarIntConstantExpr &y) {
                        Put('('), Walk(y), Put(')');
                      },
                      [&](const Star &) { Put("(*)"); },
                      [&](const Default &) { Word("DEFAULT"); },
                  },
        std::get<SelectRankCaseStmt::Rank>(x.t).u);
    Walk(" ", std::get<std::optional<Name>>(x.t)), Indent();
  }
  void Unparse(const SelectTypeStmt &x) { // R1153
    Walk(std::get<0>(x.t), ": ");
    Word("SELECT TYPE ("), Walk(std::get<1>(x.t), " => ");
    Walk(std::get<Selector>(x.t)), Put(')'), Indent();
  }
  void Unparse(const TypeGuardStmt &x) { // R1154
    Outdent(), Walk(std::get<TypeGuardStmt::Guard>(x.t));
    Walk(" ", std::get<std::optional<Name>>(x.t)), Indent();
  }
  void Unparse(const TypeGuardStmt::Guard &x) {
    common::visit(
        common::visitors{
            [&](const TypeSpec &y) { Word("TYPE IS ("), Walk(y), Put(')'); },
            [&](const DerivedTypeSpec &y) {
              Word("CLASS IS ("), Walk(y), Put(')');
            },
            [&](const Default &) { Word("CLASS DEFAULT"); },
        },
        x.u);
  }
  void Unparse(const ExitStmt &x) { // R1156
    Word("EXIT"), Walk(" ", x.v);
  }
  void Before(const GotoStmt &) { // R1157
    Word("GO TO ");
  }
  void Unparse(const ComputedGotoStmt &x) { // R1158
    Word("GO TO ("), Walk(x.t, "), ");
  }
  void Unparse(const ContinueStmt &) { // R1159
    Word("CONTINUE");
  }
  void Unparse(const StopStmt &x) { // R1160, R1161
    if (std::get<StopStmt::Kind>(x.t) == StopStmt::Kind::ErrorStop) {
      Word("ERROR ");
    }
    Word("STOP"), Walk(" ", std::get<std::optional<StopCode>>(x.t));
    Walk(", QUIET=", std::get<std::optional<ScalarLogicalExpr>>(x.t));
  }
  void Unparse(const FailImageStmt &) { // R1163
    Word("FAIL IMAGE");
  }
  void Unparse(const NotifyWaitStmt &x) { // F2023: R1166
    Word("NOTIFY WAIT ("), Walk(std::get<Scalar<Variable>>(x.t));
    Walk(", ", std::get<std::list<EventWaitSpec>>(x.t), ", ");
    Put(')');
  }
  void Unparse(const SyncAllStmt &x) { // R1164
    Word("SYNC ALL ("), Walk(x.v, ", "), Put(')');
  }
  void Unparse(const SyncImagesStmt &x) { // R1166
    Word("SYNC IMAGES (");
    Walk(std::get<SyncImagesStmt::ImageSet>(x.t));
    Walk(", ", std::get<std::list<StatOrErrmsg>>(x.t), ", "), Put(')');
  }
  void Unparse(const SyncMemoryStmt &x) { // R1168
    Word("SYNC MEMORY ("), Walk(x.v, ", "), Put(')');
  }
  void Unparse(const SyncTeamStmt &x) { // R1169
    Word("SYNC TEAM ("), Walk(std::get<TeamValue>(x.t));
    Walk(", ", std::get<std::list<StatOrErrmsg>>(x.t), ", "), Put(')');
  }
  void Unparse(const EventPostStmt &x) { // R1170
    Word("EVENT POST ("), Walk(std::get<EventVariable>(x.t));
    Walk(", ", std::get<std::list<StatOrErrmsg>>(x.t), ", "), Put(')');
  }
  void Before(const EventWaitSpec &x) { // R1173, R1174
    common::visit(common::visitors{
                      [&](const ScalarIntExpr &) { Word("UNTIL_COUNT="); },
                      [](const StatOrErrmsg &) {},
                  },
        x.u);
  }
  void Unparse(const EventWaitStmt &x) { // R1170
    Word("EVENT WAIT ("), Walk(std::get<EventVariable>(x.t));
    Walk(", ", std::get<std::list<EventWaitSpec>>(x.t), ", ");
    Put(')');
  }
  void Unparse(const FormTeamStmt &x) { // R1175, R1177
    Word("FORM TEAM ("), Walk(std::get<ScalarIntExpr>(x.t));
    Put(','), Walk(std::get<TeamVariable>(x.t));
    Walk(", ", std::get<std::list<FormTeamStmt::FormTeamSpec>>(x.t), ", ");
    Put(')');
  }
  void Before(const FormTeamStmt::FormTeamSpec &x) { // R1176, R1178
    common::visit(common::visitors{
                      [&](const ScalarIntExpr &) { Word("NEW_INDEX="); },
                      [](const StatOrErrmsg &) {},
                  },
        x.u);
  }
  void Unparse(const LockStmt &x) { // R1179
    Word("LOCK ("), Walk(std::get<LockVariable>(x.t));
    Walk(", ", std::get<std::list<LockStmt::LockStat>>(x.t), ", ");
    Put(')');
  }
  void Before(const LockStmt::LockStat &x) { // R1180
    common::visit(
        common::visitors{
            [&](const ScalarLogicalVariable &) { Word("ACQUIRED_LOCK="); },
            [](const StatOrErrmsg &) {},
        },
        x.u);
  }
  void Unparse(const UnlockStmt &x) { // R1181
    Word("UNLOCK ("), Walk(std::get<LockVariable>(x.t));
    Walk(", ", std::get<std::list<StatOrErrmsg>>(x.t), ", ");
    Put(')');
  }

  void Unparse(const OpenStmt &x) { // R1204
    Word("OPEN ("), Walk(x.v, ", "), Put(')');
  }
  bool Pre(const ConnectSpec &x) { // R1205
    return common::visit(common::visitors{
                             [&](const FileUnitNumber &) {
                               Word("UNIT=");
                               return true;
                             },
                             [&](const FileNameExpr &) {
                               Word("FILE=");
                               return true;
                             },
                             [&](const ConnectSpec::CharExpr &y) {
                               Walk(y.t, "=");
                               return false;
                             },
                             [&](const MsgVariable &) {
                               Word("IOMSG=");
                               return true;
                             },
                             [&](const StatVariable &) {
                               Word("IOSTAT=");
                               return true;
                             },
                             [&](const ConnectSpec::Recl &) {
                               Word("RECL=");
                               return true;
                             },
                             [&](const ConnectSpec::Newunit &) {
                               Word("NEWUNIT=");
                               return true;
                             },
                             [&](const ErrLabel &) {
                               Word("ERR=");
                               return true;
                             },
                             [&](const StatusExpr &) {
                               Word("STATUS=");
                               return true;
                             },
                         },
        x.u);
  }
  void Unparse(const CloseStmt &x) { // R1208
    Word("CLOSE ("), Walk(x.v, ", "), Put(')');
  }
  void Before(const CloseStmt::CloseSpec &x) { // R1209
    common::visit(common::visitors{
                      [&](const FileUnitNumber &) { Word("UNIT="); },
                      [&](const StatVariable &) { Word("IOSTAT="); },
                      [&](const MsgVariable &) { Word("IOMSG="); },
                      [&](const ErrLabel &) { Word("ERR="); },
                      [&](const StatusExpr &) { Word("STATUS="); },
                  },
        x.u);
  }
  void Unparse(const ReadStmt &x) { // R1210
    Word("READ ");
    if (x.iounit) {
      Put('('), Walk(x.iounit);
      if (x.format) {
        Put(", "), Walk(x.format);
      }
      Walk(", ", x.controls, ", ");
      Put(')');
    } else if (x.format) {
      Walk(x.format);
      if (!x.items.empty()) {
        Put(", ");
      }
    } else {
      Put('('), Walk(x.controls, ", "), Put(')');
    }
    Walk(" ", x.items, ", ");
  }
  void Unparse(const WriteStmt &x) { // R1211
    Word("WRITE (");
    if (x.iounit) {
      Walk(x.iounit);
      if (x.format) {
        Put(", "), Walk(x.format);
      }
      Walk(", ", x.controls, ", ");
    } else {
      Walk(x.controls, ", ");
    }
    Put(')'), Walk(" ", x.items, ", ");
  }
  void Unparse(const PrintStmt &x) { // R1212
    Word("PRINT "), Walk(std::get<Format>(x.t));
    Walk(", ", std::get<std::list<OutputItem>>(x.t), ", ");
  }
  bool Pre(const IoControlSpec &x) { // R1213
    return common::visit(common::visitors{
                             [&](const IoUnit &) {
                               Word("UNIT=");
                               return true;
                             },
                             [&](const Format &) {
                               Word("FMT=");
                               return true;
                             },
                             [&](const Name &) {
                               Word("NML=");
                               return true;
                             },
                             [&](const IoControlSpec::CharExpr &y) {
                               Walk(y.t, "=");
                               return false;
                             },
                             [&](const IoControlSpec::Asynchronous &) {
                               Word("ASYNCHRONOUS=");
                               return true;
                             },
                             [&](const EndLabel &) {
                               Word("END=");
                               return true;
                             },
                             [&](const EorLabel &) {
                               Word("EOR=");
                               return true;
                             },
                             [&](const ErrLabel &) {
                               Word("ERR=");
                               return true;
                             },
                             [&](const IdVariable &) {
                               Word("ID=");
                               return true;
                             },
                             [&](const MsgVariable &) {
                               Word("IOMSG=");
                               return true;
                             },
                             [&](const StatVariable &) {
                               Word("IOSTAT=");
                               return true;
                             },
                             [&](const IoControlSpec::Pos &) {
                               Word("POS=");
                               return true;
                             },
                             [&](const IoControlSpec::Rec &) {
                               Word("REC=");
                               return true;
                             },
                             [&](const IoControlSpec::Size &) {
                               Word("SIZE=");
                               return true;
                             },
                         },
        x.u);
  }
  void Unparse(const InputImpliedDo &x) { // R1218
    Put('('), Walk(std::get<std::list<InputItem>>(x.t), ", "), Put(", ");
    Walk(std::get<IoImpliedDoControl>(x.t)), Put(')');
  }
  void Unparse(const OutputImpliedDo &x) { // R1219
    Put('('), Walk(std::get<std::list<OutputItem>>(x.t), ", "), Put(", ");
    Walk(std::get<IoImpliedDoControl>(x.t)), Put(')');
  }
  void Unparse(const WaitStmt &x) { // R1222
    Word("WAIT ("), Walk(x.v, ", "), Put(')');
  }
  void Before(const WaitSpec &x) { // R1223
    common::visit(common::visitors{
                      [&](const FileUnitNumber &) { Word("UNIT="); },
                      [&](const EndLabel &) { Word("END="); },
                      [&](const EorLabel &) { Word("EOR="); },
                      [&](const ErrLabel &) { Word("ERR="); },
                      [&](const IdExpr &) { Word("ID="); },
                      [&](const MsgVariable &) { Word("IOMSG="); },
                      [&](const StatVariable &) { Word("IOSTAT="); },
                  },
        x.u);
  }
  void Unparse(const BackspaceStmt &x) { // R1224
    Word("BACKSPACE ("), Walk(x.v, ", "), Put(')');
  }
  void Unparse(const EndfileStmt &x) { // R1225
    Word("ENDFILE ("), Walk(x.v, ", "), Put(')');
  }
  void Unparse(const RewindStmt &x) { // R1226
    Word("REWIND ("), Walk(x.v, ", "), Put(')');
  }
  void Before(const PositionOrFlushSpec &x) { // R1227 & R1229
    common::visit(common::visitors{
                      [&](const FileUnitNumber &) { Word("UNIT="); },
                      [&](const MsgVariable &) { Word("IOMSG="); },
                      [&](const StatVariable &) { Word("IOSTAT="); },
                      [&](const ErrLabel &) { Word("ERR="); },
                  },
        x.u);
  }
  void Unparse(const FlushStmt &x) { // R1228
    Word("FLUSH ("), Walk(x.v, ", "), Put(')');
  }
  void Unparse(const InquireStmt &x) { // R1230
    Word("INQUIRE (");
    common::visit(
        common::visitors{
            [&](const InquireStmt::Iolength &y) {
              Word("IOLENGTH="), Walk(y.t, ") ");
            },
            [&](const std::list<InquireSpec> &y) { Walk(y, ", "), Put(')'); },
        },
        x.u);
  }
  bool Pre(const InquireSpec &x) { // R1231
    return common::visit(common::visitors{
                             [&](const FileUnitNumber &) {
                               Word("UNIT=");
                               return true;
                             },
                             [&](const FileNameExpr &) {
                               Word("FILE=");
                               return true;
                             },
                             [&](const InquireSpec::CharVar &y) {
                               Walk(y.t, "=");
                               return false;
                             },
                             [&](const InquireSpec::IntVar &y) {
                               Walk(y.t, "=");
                               return false;
                             },
                             [&](const InquireSpec::LogVar &y) {
                               Walk(y.t, "=");
                               return false;
                             },
                             [&](const IdExpr &) {
                               Word("ID=");
                               return true;
                             },
                             [&](const ErrLabel &) {
                               Word("ERR=");
                               return true;
                             },
                         },
        x.u);
  }

  void Before(const FormatStmt &) { // R1301
    Word("FORMAT");
  }
  void Unparse(const format::FormatSpecification &x) { // R1302, R1303, R1305
    Put('('), Walk("", x.items, ",", x.unlimitedItems.empty() ? "" : ",");
    Walk("*(", x.unlimitedItems, ",", ")"), Put(')');
  }
  void Unparse(const format::FormatItem &x) { // R1304, R1306, R1321
    if (x.repeatCount) {
      Walk(*x.repeatCount);
    }
    common::visit(common::visitors{
                      [&](const std::string &y) { PutNormalized(y); },
                      [&](const std::list<format::FormatItem> &y) {
                        Walk("(", y, ",", ")");
                      },
                      [&](const auto &y) { Walk(y); },
                  },
        x.u);
  }
  void Unparse(
      const format::IntrinsicTypeDataEditDesc &x) { // R1307(1/2) - R1311
    switch (x.kind) {
#define FMT(x) \
  case format::IntrinsicTypeDataEditDesc::Kind::x: \
    Put(#x); \
    break
      FMT(I);
      FMT(B);
      FMT(O);
      FMT(Z);
      FMT(F);
      FMT(E);
      FMT(EN);
      FMT(ES);
      FMT(EX);
      FMT(G);
      FMT(L);
      FMT(A);
      FMT(D);
#undef FMT
    }
    Walk(x.width), Walk(".", x.digits), Walk("E", x.exponentWidth);
  }
  void Unparse(const format::DerivedTypeDataEditDesc &x) { // R1307(2/2), R1312
    Word("DT");
    if (!x.type.empty()) {
      Put('"'), Put(x.type), Put('"');
    }
    Walk("(", x.parameters, ",", ")");
  }
  void Unparse(const format::ControlEditDesc &x) { // R1313, R1315-R1320
    switch (x.kind) {
    case format::ControlEditDesc::Kind::T:
      Word("T");
      Walk(x.count);
      break;
    case format::ControlEditDesc::Kind::TL:
      Word("TL");
      Walk(x.count);
      break;
    case format::ControlEditDesc::Kind::TR:
      Word("TR");
      Walk(x.count);
      break;
    case format::ControlEditDesc::Kind::X:
      if (x.count != 1) {
        Walk(x.count);
      }
      Word("X");
      break;
    case format::ControlEditDesc::Kind::Slash:
      if (x.count != 1) {
        Walk(x.count);
      }
      Put('/');
      break;
    case format::ControlEditDesc::Kind::Colon:
      Put(':');
      break;
    case format::ControlEditDesc::Kind::P:
      Walk(x.count);
      Word("P");
      break;
#define FMT(x) \
  case format::ControlEditDesc::Kind::x: \
    Put(#x); \
    break
      FMT(SS);
      FMT(SP);
      FMT(S);
      FMT(BN);
      FMT(BZ);
      FMT(RU);
      FMT(RD);
      FMT(RZ);
      FMT(RN);
      FMT(RC);
      FMT(RP);
      FMT(DC);
      FMT(DP);
#undef FMT
    case format::ControlEditDesc::Kind::Dollar:
      Put('$');
      break;
    case format::ControlEditDesc::Kind::Backslash:
      Put('\\');
      break;
    }
  }

  void Before(const MainProgram &x) { // R1401
    if (!std::get<std::optional<Statement<ProgramStmt>>>(x.t)) {
      Indent();
    }
  }
  void Before(const ProgramStmt &) { // R1402
    Word("PROGRAM "), Indent();
  }
  void Unparse(const EndProgramStmt &x) { // R1403
    EndSubprogram("PROGRAM", x.v);
  }
  void Before(const ModuleStmt &) { // R1405
    Word("MODULE "), Indent();
  }
  void Unparse(const EndModuleStmt &x) { // R1406
    EndSubprogram("MODULE", x.v);
  }
  void Unparse(const UseStmt &x) { // R1409
    Word("USE"), Walk(", ", x.nature), Put(" :: "), Walk(x.moduleName);
    common::visit(
        common::visitors{
            [&](const std::list<Rename> &y) { Walk(", ", y, ", "); },
            [&](const std::list<Only> &y) { Walk(", ONLY: ", y, ", "); },
        },
        x.u);
  }
  void Unparse(const Rename &x) { // R1411
    common::visit(common::visitors{
                      [&](const Rename::Names &y) { Walk(y.t, " => "); },
                      [&](const Rename::Operators &y) {
                        Word("OPERATOR("), Walk(y.t, ") => OPERATOR("),
                            Put(")");
                      },
                  },
        x.u);
  }
  void Unparse(const SubmoduleStmt &x) { // R1417
    Word("SUBMODULE ("), WalkTupleElements(x.t, ")"), Indent();
  }
  void Unparse(const ParentIdentifier &x) { // R1418
    Walk(std::get<Name>(x.t)), Walk(":", std::get<std::optional<Name>>(x.t));
  }
  void Unparse(const EndSubmoduleStmt &x) { // R1419
    EndSubprogram("SUBMODULE", x.v);
  }
  void Unparse(const BlockDataStmt &x) { // R1421
    Word("BLOCK DATA"), Walk(" ", x.v), Indent();
  }
  void Unparse(const EndBlockDataStmt &x) { // R1422
    EndSubprogram("BLOCK DATA", x.v);
  }

  void Unparse(const InterfaceStmt &x) { // R1503
    common::visit(common::visitors{
                      [&](const std::optional<GenericSpec> &y) {
                        Word("INTERFACE"), Walk(" ", y);
                      },
                      [&](const Abstract &) { Word("ABSTRACT INTERFACE"); },
                  },
        x.u);
    Indent();
  }
  void Unparse(const EndInterfaceStmt &x) { // R1504
    Outdent(), Word("END INTERFACE"), Walk(" ", x.v);
  }
  void Unparse(const ProcedureStmt &x) { // R1506
    if (std::get<ProcedureStmt::Kind>(x.t) ==
        ProcedureStmt::Kind::ModuleProcedure) {
      Word("MODULE ");
    }
    Word("PROCEDURE :: ");
    Walk(std::get<std::list<Name>>(x.t), ", ");
  }
  void Before(const GenericSpec &x) { // R1508, R1509
    common::visit(
        common::visitors{
            [&](const DefinedOperator &) { Word("OPERATOR("); },
            [&](const GenericSpec::Assignment &) { Word("ASSIGNMENT(=)"); },
            [&](const GenericSpec::ReadFormatted &) {
              Word("READ(FORMATTED)");
            },
            [&](const GenericSpec::ReadUnformatted &) {
              Word("READ(UNFORMATTED)");
            },
            [&](const GenericSpec::WriteFormatted &) {
              Word("WRITE(FORMATTED)");
            },
            [&](const GenericSpec::WriteUnformatted &) {
              Word("WRITE(UNFORMATTED)");
            },
            [](const auto &) {},
        },
        x.u);
  }
  void Post(const GenericSpec &x) {
    common::visit(common::visitors{
                      [&](const DefinedOperator &) { Put(')'); },
                      [](const auto &) {},
                  },
        x.u);
  }
  void Unparse(const GenericStmt &x) { // R1510
    Word("GENERIC"), Walk(", ", std::get<std::optional<AccessSpec>>(x.t));
    Put(" :: "), Walk(std::get<GenericSpec>(x.t)), Put(" => ");
    Walk(std::get<std::list<Name>>(x.t), ", ");
  }
  void Unparse(const ExternalStmt &x) { // R1511
    Word("EXTERNAL :: "), Walk(x.v, ", ");
  }
  void Unparse(const ProcedureDeclarationStmt &x) { // R1512
    Word("PROCEDURE("), Walk(std::get<std::optional<ProcInterface>>(x.t));
    Put(')'), Walk(", ", std::get<std::list<ProcAttrSpec>>(x.t), ", ");
    Put(" :: "), Walk(std::get<std::list<ProcDecl>>(x.t), ", ");
  }
  void Unparse(const ProcDecl &x) { // R1515
    Walk(std::get<Name>(x.t));
    Walk(" => ", std::get<std::optional<ProcPointerInit>>(x.t));
  }
  void Unparse(const IntrinsicStmt &x) { // R1519
    Word("INTRINSIC :: "), Walk(x.v, ", ");
  }
  void Unparse(const CallStmt::StarOrExpr &x) {
    if (x.v) {
      Walk(*x.v);
    } else {
      Word("*");
    }
  }
  void Unparse(const CallStmt::Chevrons &x) { // CUDA
    Walk(std::get<0>(x.t)); // grid
    Word(","), Walk(std::get<1>(x.t)); // block
    Walk(",", std::get<2>(x.t)); // bytes
    Walk(",", std::get<3>(x.t)); // stream
  }
  void Unparse(const FunctionReference &x) { // R1520
    Walk(std::get<ProcedureDesignator>(x.v.t));
    Put('('), Walk(std::get<std::list<ActualArgSpec>>(x.v.t), ", "), Put(')');
  }
  void Unparse(const CallStmt &x) { // R1521
    if (asFortran_ && x.typedCall.get()) {
      Put(' ');
      asFortran_->call(out_, *x.typedCall);
      Put('\n');
    } else {
      const auto &pd{std::get<ProcedureDesignator>(x.call.t)};
      Word("CALL "), Walk(pd);
      Walk("<<<", x.chevrons, ">>>");
      const auto &args{std::get<std::list<ActualArgSpec>>(x.call.t)};
      if (args.empty()) {
        if (std::holds_alternative<ProcComponentRef>(pd.u)) {
          Put("()"); // pgf90 crashes on CALL to tbp without parentheses
        }
      } else {
        Walk("(", args, ", ", ")");
      }
    }
  }
  void Unparse(const ActualArgSpec &x) { // R1523
    Walk(std::get<std::optional<Keyword>>(x.t), "=");
    Walk(std::get<ActualArg>(x.t));
  }
  void Unparse(const ActualArg::PercentRef &x) { // R1524
    Word("%REF("), Walk(x.v), Put(')');
  }
  void Unparse(const ActualArg::PercentVal &x) {
    Word("%VAL("), Walk(x.v), Put(')');
  }
  void Before(const AltReturnSpec &) { // R1525
    Put('*');
  }
  void Post(const PrefixSpec::Elemental) { Word("ELEMENTAL"); } // R1527
  void Post(const PrefixSpec::Impure) { Word("IMPURE"); }
  void Post(const PrefixSpec::Module) { Word("MODULE"); }
  void Post(const PrefixSpec::Non_Recursive) { Word("NON_RECURSIVE"); }
  void Post(const PrefixSpec::Pure) { Word("PURE"); }
  void Post(const PrefixSpec::Recursive) { Word("RECURSIVE"); }
  void Unparse(const PrefixSpec::Attributes &x) {
    Word("ATTRIBUTES("), Walk(x.v), Word(")");
  }
  void Unparse(const PrefixSpec::Launch_Bounds &x) {
    Word("LAUNCH_BOUNDS("), Walk(x.v), Word(")");
  }
  void Unparse(const PrefixSpec::Cluster_Dims &x) {
    Word("CLUSTER_DIMS("), Walk(x.v), Word(")");
  }
  void Unparse(const FunctionStmt &x) { // R1530
    Walk("", std::get<std::list<PrefixSpec>>(x.t), " ", " ");
    Word("FUNCTION "), Walk(std::get<Name>(x.t)), Put("(");
    Walk(std::get<std::list<Name>>(x.t), ", "), Put(')');
    Walk(" ", std::get<std::optional<Suffix>>(x.t)), Indent();
  }
  void Unparse(const Suffix &x) { // R1532
    if (x.resultName) {
      Word("RESULT("), Walk(x.resultName), Put(')');
      Walk(" ", x.binding);
    } else {
      Walk(x.binding);
    }
  }
  void Unparse(const EndFunctionStmt &x) { // R1533
    EndSubprogram("FUNCTION", x.v);
  }
  void Unparse(const SubroutineStmt &x) { // R1535
    Walk("", std::get<std::list<PrefixSpec>>(x.t), " ", " ");
    Word("SUBROUTINE "), Walk(std::get<Name>(x.t));
    const auto &args{std::get<std::list<DummyArg>>(x.t)};
    const auto &bind{std::get<std::optional<LanguageBindingSpec>>(x.t)};
    if (args.empty()) {
      Walk(" () ", bind);
    } else {
      Walk(" (", args, ", ", ")");
      Walk(" ", bind);
    }
    Indent();
  }
  void Unparse(const EndSubroutineStmt &x) { // R1537
    EndSubprogram("SUBROUTINE", x.v);
  }
  void Before(const MpSubprogramStmt &) { // R1539
    Word("MODULE PROCEDURE "), Indent();
  }
  void Unparse(const EndMpSubprogramStmt &x) { // R1540
    EndSubprogram("PROCEDURE", x.v);
  }
  void Unparse(const EntryStmt &x) { // R1541
    Word("ENTRY "), Walk(std::get<Name>(x.t)), Put("(");
    Walk(std::get<std::list<DummyArg>>(x.t), ", "), Put(")");
    Walk(" ", std::get<std::optional<Suffix>>(x.t));
  }
  void Unparse(const ReturnStmt &x) { // R1542
    Word("RETURN"), Walk(" ", x.v);
  }
  void Unparse(const ContainsStmt &) { // R1543
    Outdent();
    Word("CONTAINS");
    Indent();
  }
  void Unparse(const StmtFunctionStmt &x) { // R1544
    Walk(std::get<Name>(x.t)), Put('(');
    Walk(std::get<std::list<Name>>(x.t), ", "), Put(") = ");
    Walk(std::get<Scalar<Expr>>(x.t));
  }

  // Directives, extensions, and deprecated constructs
  void Unparse(const CompilerDirective &x) {
    common::visit(
        common::visitors{
            [&](const std::list<CompilerDirective::IgnoreTKR> &tkr) {
              Word("!DIR$ IGNORE_TKR"); // emitted even if tkr list is empty
              Walk(" ", tkr, ", ");
            },
            [&](const CompilerDirective::LoopCount &lcount) {
              Walk("!DIR$ LOOP COUNT (", lcount.v, ", ", ")");
            },
            [&](const std::list<CompilerDirective::AssumeAligned>
                    &assumeAligned) {
              Word("!DIR$ ASSUME_ALIGNED ");
              Walk(" ", assumeAligned, ", ");
            },
            [&](const CompilerDirective::VectorAlways &valways) {
              Word("!DIR$ VECTOR ALWAYS");
            },
            [&](const std::list<CompilerDirective::NameValue> &names) {
              Walk("!DIR$ ", names, " ");
            },
            [&](const CompilerDirective::Unroll &unroll) {
              Word("!DIR$ UNROLL");
              Walk(" ", unroll.v);
            },
            [&](const CompilerDirective::UnrollAndJam &unrollAndJam) {
              Word("!DIR$ UNROLL_AND_JAM");
              Walk(" ", unrollAndJam.v);
            },
            [&](const CompilerDirective::NoVector &) {
              Word("!DIR$ NOVECTOR");
            },
            [&](const CompilerDirective::NoUnroll &) {
              Word("!DIR$ NOUNROLL");
            },
            [&](const CompilerDirective::NoUnrollAndJam &) {
              Word("!DIR$ NOUNROLL_AND_JAM");
            },
            [&](const CompilerDirective::Unrecognized &) {
              Word("!DIR$ ");
              Word(x.source.ToString());
            },
        },
        x.u);
    Put('\n');
  }
  void Unparse(const CompilerDirective::IgnoreTKR &x) {
    if (const auto &maybeList{
            std::get<std::optional<std::list<const char *>>>(x.t)}) {
      Put("(");
      for (const char *tkr : *maybeList) {
        Put(*tkr);
      }
      Put(") ");
    }
    Walk(std::get<Name>(x.t));
  }
  void Unparse(const CompilerDirective::NameValue &x) {
    Walk(std::get<Name>(x.t));
    Walk("=", std::get<std::optional<std::uint64_t>>(x.t));
  }
  void Unparse(const CompilerDirective::AssumeAligned &x) {
    Walk(std::get<common::Indirection<Designator>>(x.t));
    Put(":");
    Walk(std::get<uint64_t>(x.t));
  }

  // OpenACC Directives & Clauses
  void Unparse(const AccAtomicCapture &x) {
    BeginOpenACC();
    Word("!$ACC CAPTURE");
    Put("\n");
    EndOpenACC();
    Walk(std::get<AccAtomicCapture::Stmt1>(x.t));
    Put("\n");
    Walk(std::get<AccAtomicCapture::Stmt2>(x.t));
    BeginOpenACC();
    Word("!$ACC END ATOMIC\n");
    EndOpenACC();
  }
  void Unparse(const AccAtomicRead &x) {
    BeginOpenACC();
    Word("!$ACC ATOMIC READ");
    Put("\n");
    EndOpenACC();
    Walk(std::get<Statement<AssignmentStmt>>(x.t));
    BeginOpenACC();
    Walk(std::get<std::optional<AccEndAtomic>>(x.t), "!$ACC END ATOMIC\n");
    EndOpenACC();
  }
  void Unparse(const AccAtomicWrite &x) {
    BeginOpenACC();
    Word("!$ACC ATOMIC WRITE");
    Put("\n");
    EndOpenACC();
    Walk(std::get<Statement<AssignmentStmt>>(x.t));
    BeginOpenACC();
    Walk(std::get<std::optional<AccEndAtomic>>(x.t), "!$ACC END ATOMIC\n");
    EndOpenACC();
  }
  void Unparse(const AccAtomicUpdate &x) {
    BeginOpenACC();
    Word("!$ACC ATOMIC UPDATE");
    Put("\n");
    EndOpenACC();
    Walk(std::get<Statement<AssignmentStmt>>(x.t));
    BeginOpenACC();
    Walk(std::get<std::optional<AccEndAtomic>>(x.t), "!$ACC END ATOMIC\n");
    EndOpenACC();
  }
  void Unparse(const llvm::acc::Directive &x) {
    Word(llvm::acc::getOpenACCDirectiveName(x).str());
  }
#define GEN_FLANG_CLAUSE_UNPARSE
#include "llvm/Frontend/OpenACC/ACC.inc"
  void Unparse(const AccObjectListWithModifier &x) {
    Walk(std::get<std::optional<AccDataModifier>>(x.t), ":");
    Walk(std::get<AccObjectList>(x.t));
  }
  void Unparse(const AccBindClause &x) {
    common::visit(common::visitors{
                      [&](const Name &y) { Walk(y); },
                      [&](const ScalarDefaultCharExpr &y) { Walk(y); },
                  },
        x.u);
  }
  void Unparse(const AccDefaultClause &x) {
    switch (x.v) {
    case llvm::acc::DefaultValue::ACC_Default_none:
      Put("NONE");
      break;
    case llvm::acc::DefaultValue::ACC_Default_present:
      Put("PRESENT");
      break;
    }
  }
  void Unparse(const AccClauseList &x) { Walk(" ", x.v, " "); }
  void Unparse(const AccGangArgList &x) { Walk(x.v, ","); }
  void Before(const AccSizeExpr &x) {
    if (!x.v)
      Put("*");
  }
  void Before(const AccGangArg &x) {
    common::visit(common::visitors{
                      [&](const AccGangArg::Num &) { Word("NUM:"); },
                      [&](const AccGangArg::Dim &) { Word("DIM:"); },
                      [&](const AccGangArg::Static &) { Word("STATIC:"); },
                      [](const StatOrErrmsg &) {},
                  },
        x.u);
  }
  void Unparse(const AccCollapseArg &x) {
    const auto &force{std::get<bool>(x.t)};
    const auto &collapseValue{std::get<parser::ScalarIntConstantExpr>(x.t)};
    if (force) {
      Put("FORCE:");
    }
    Walk(collapseValue);
  }
  void Unparse(const OpenACCBlockConstruct &x) {
    BeginOpenACC();
    Word("!$ACC ");
    Walk(std::get<AccBeginBlockDirective>(x.t));
    Put("\n");
    EndOpenACC();
    Walk(std::get<Block>(x.t), "");
    BeginOpenACC();
    Word("!$ACC END ");
    Walk(std::get<AccEndBlockDirective>(x.t));
    Put("\n");
    EndOpenACC();
  }
  void Unparse(const OpenACCLoopConstruct &x) {
    BeginOpenACC();
    Word("!$ACC ");
    Walk(std::get<AccBeginLoopDirective>(x.t));
    Put("\n");
    EndOpenACC();
    Walk(std::get<std::optional<DoConstruct>>(x.t));
  }
  void Unparse(const AccBeginLoopDirective &x) {
    Walk(std::get<AccLoopDirective>(x.t));
    Walk(std::get<AccClauseList>(x.t));
  }
  void Unparse(const OpenACCStandaloneConstruct &x) {
    BeginOpenACC();
    Word("!$ACC ");
    Walk(std::get<AccStandaloneDirective>(x.t));
    Walk(std::get<AccClauseList>(x.t));
    Put("\n");
    EndOpenACC();
  }
  void Unparse(const OpenACCStandaloneDeclarativeConstruct &x) {
    BeginOpenACC();
    Word("!$ACC ");
    Walk(std::get<AccDeclarativeDirective>(x.t));
    Walk(std::get<AccClauseList>(x.t));
    Put("\n");
    EndOpenACC();
  }
  void Unparse(const OpenACCCombinedConstruct &x) {
    BeginOpenACC();
    Word("!$ACC ");
    Walk(std::get<AccBeginCombinedDirective>(x.t));
    Put("\n");
    EndOpenACC();
    Walk(std::get<std::optional<DoConstruct>>(x.t));
    BeginOpenACC();
    Walk("!$ACC END ", std::get<std::optional<AccEndCombinedDirective>>(x.t),
        "\n");
    EndOpenACC();
  }
  void Unparse(const OpenACCRoutineConstruct &x) {
    BeginOpenACC();
    Word("!$ACC ROUTINE");
    Walk("(", std::get<std::optional<Name>>(x.t), ")");
    Walk(std::get<AccClauseList>(x.t));
    Put("\n");
    EndOpenACC();
  }
  void Unparse(const AccObject &x) {
    common::visit(common::visitors{
                      [&](const Designator &y) { Walk(y); },
                      [&](const Name &y) { Put("/"), Walk(y), Put("/"); },
                  },
        x.u);
  }
  void Unparse(const AccObjectList &x) { Walk(x.v, ","); }
  void Unparse(const AccObjectListWithReduction &x) {
    Walk(std::get<ReductionOperator>(x.t));
    Put(":");
    Walk(std::get<AccObjectList>(x.t));
  }
  void Unparse(const OpenACCCacheConstruct &x) {
    BeginOpenACC();
    Word("!$ACC ");
    Word("CACHE(");
    Walk(std::get<AccObjectListWithModifier>(x.t));
    Put(")");
    Put("\n");
    EndOpenACC();
  }
  void Unparse(const AccWaitArgument &x) {
    Walk("DEVNUM:", std::get<std::optional<ScalarIntExpr>>(x.t), ":");
    Walk(std::get<std::list<ScalarIntExpr>>(x.t), ",");
  }
  void Unparse(const OpenACCWaitConstruct &x) {
    BeginOpenACC();
    Word("!$ACC ");
    Word("WAIT(");
    Walk(std::get<std::optional<AccWaitArgument>>(x.t));
    Walk(std::get<AccClauseList>(x.t));
    Put(")");
    Put("\n");
    EndOpenACC();
  }

  // OpenMP Clauses & Directives
  void Unparse(const OmpArgumentList &x) { Walk(x.v, ", "); }

  void Unparse(const OmpTypeNameList &x) { //
    Walk(x.v, ",");
  }
  void Unparse(const OmpMapperSpecifier &x) {
    const auto &mapperName{std::get<std::string>(x.t)};
    if (mapperName.find(llvm::omp::OmpDefaultMapperName) == std::string::npos) {
      Walk(mapperName);
      Put(":");
    }
    Walk(std::get<TypeSpec>(x.t));
    Put("::");
    Walk(std::get<Name>(x.t));
  }
  void Unparse(const OmpReductionSpecifier &x) {
    Walk(std::get<OmpReductionIdentifier>(x.t));
    Put(":");
    Walk(std::get<OmpTypeNameList>(x.t));
    Walk(":", std::get<std::optional<OmpReductionCombiner>>(x.t));
  }
  void Unparse(const llvm::omp::Directive &x) {
    unsigned ompVersion{langOpts_.OpenMPVersion};
    Word(llvm::omp::getOpenMPDirectiveName(x, ompVersion).str());
  }
  void Unparse(const OmpDirectiveSpecification &x) {
    auto unparseArgs{[&]() {
      if (auto &args{std::get<std::optional<OmpArgumentList>>(x.t)}) {
        Put("(");
        Walk(*args);
        Put(")");
      }
    }};
    auto unparseClauses{[&]() { //
      Walk(std::get<std::optional<OmpClauseList>>(x.t));
    }};

    Walk(std::get<OmpDirectiveName>(x.t));
    auto flags{std::get<OmpDirectiveSpecification::Flags>(x.t)};
    if (flags == OmpDirectiveSpecification::Flags::DeprecatedSyntax) {
      if (x.DirId() == llvm::omp::Directive::OMPD_flush) {
        // FLUSH clause arglist
        unparseClauses();
        unparseArgs();
      }
    } else {
      unparseArgs();
      unparseClauses();
    }
  }
  void Unparse(const OmpTraitScore &x) {
    Word("SCORE(");
    Walk(x.v);
    Put(")");
  }
  void Unparse(const OmpTraitPropertyExtension::Complex &x) {
    using PropList = std::list<common::Indirection<OmpTraitPropertyExtension>>;
    Walk(std::get<OmpTraitPropertyName>(x.t));
    Put("(");
    Walk(std::get<PropList>(x.t), ",");
    Put(")");
  }
  void Unparse(const OmpTraitSelector &x) {
    Walk(std::get<OmpTraitSelectorName>(x.t));
    Walk(std::get<std::optional<OmpTraitSelector::Properties>>(x.t));
  }
  void Unparse(const OmpTraitSelector::Properties &x) {
    Put("(");
    Walk(std::get<std::optional<OmpTraitScore>>(x.t), ": ");
    Walk(std::get<std::list<OmpTraitProperty>>(x.t));
    Put(")");
  }
  void Unparse(const OmpTraitSetSelector &x) {
    Walk(std::get<OmpTraitSetSelectorName>(x.t));
    Put("={");
    Walk(std::get<std::list<OmpTraitSelector>>(x.t));
    Put("}");
  }
  void Unparse(const OmpContextSelectorSpecification &x) { Walk(x.v, ", "); }

  void Unparse(const OmpObject &x) {
    common::visit(common::visitors{
                      [&](const Designator &y) { Walk(y); },
                      [&](const Name &y) { Put("/"), Walk(y), Put("/"); },
                  },
        x.u);
  }
  void Unparse(const OmpDirectiveNameModifier &x) {
    unsigned ompVersion{langOpts_.OpenMPVersion};
    Word(llvm::omp::getOpenMPDirectiveName(x.v, ompVersion));
  }
  void Unparse(const OmpIteratorSpecifier &x) {
    Walk(std::get<TypeDeclarationStmt>(x.t));
    Put(" = ");
    Walk(std::get<SubscriptTriplet>(x.t));
  }
  void Unparse(const OmpIterator &x) {
    Word("ITERATOR(");
    Walk(x.v);
    Put(")");
  }
  void Unparse(const OmpMapper &x) {
    Word("MAPPER(");
    Walk(x.v);
    Put(")");
  }
  void Unparse(const OmpLastprivateClause &x) {
    using Modifier = OmpLastprivateClause::Modifier;
    Walk(std::get<std::optional<std::list<Modifier>>>(x.t), ": ");
    Walk(std::get<OmpObjectList>(x.t));
  }
  void Unparse(const OmpInteropPreference &x) { Walk(x.v, ","); }
  void Unparse(const OmpInitClause &x) {
    using Modifier = OmpInitClause::Modifier;
    auto &modifiers{std::get<std::optional<std::list<Modifier>>>(x.t)};
    bool isTypeStart{true};
    for (const Modifier &m : *modifiers) {
      if (auto *interopPreferenceMod{
              std::get_if<parser::OmpInteropPreference>(&m.u)}) {
        Put("PREFER_TYPE(");
        Walk(*interopPreferenceMod);
        Put("),");
      } else if (auto *interopTypeMod{
                     std::get_if<parser::OmpInteropType>(&m.u)}) {
        if (isTypeStart) {
          isTypeStart = false;
        } else {
          Put(",");
        }
        Walk(*interopTypeMod);
      }
    }
    Put(": ");
    Walk(std::get<OmpObject>(x.t));
  }
  void Unparse(const OmpMapClause &x) {
    using Modifier = OmpMapClause::Modifier;
    Walk(std::get<std::optional<std::list<Modifier>>>(x.t), ": ");
    Walk(std::get<OmpObjectList>(x.t));
  }
  void Unparse(const OmpScheduleClause &x) {
    using Modifier = OmpScheduleClause::Modifier;
    Walk(std::get<std::optional<std::list<Modifier>>>(x.t), ":");
    Walk(std::get<OmpScheduleClause::Kind>(x.t));
    Walk(",", std::get<std::optional<ScalarIntExpr>>(x.t));
  }
  void Unparse(const OmpDeviceClause &x) {
    using Modifier = OmpDeviceClause::Modifier;
    Walk(std::get<std::optional<std::list<Modifier>>>(x.t), ": ");
    Walk(std::get<ScalarIntExpr>(x.t));
  }
  void Unparse(const OmpAbsentClause &x) { Walk("", x.v, ","); }
  void Unparse(const OmpContainsClause &x) { Walk("", x.v, ","); }
  void Unparse(const OmpAffinityClause &x) {
    using Modifier = OmpAffinityClause::Modifier;
    Walk(std::get<std::optional<std::list<Modifier>>>(x.t), ": ");
    Walk(std::get<OmpObjectList>(x.t));
  }
  void Unparse(const OmpAlignedClause &x) {
    using Modifier = OmpAlignedClause::Modifier;
    Walk(std::get<OmpObjectList>(x.t));
    Walk(": ", std::get<std::optional<std::list<Modifier>>>(x.t));
  }
  void Unparse(const OmpDynGroupprivateClause &x) {
    using Modifier = OmpDynGroupprivateClause::Modifier;
    Walk(std::get<std::optional<std::list<Modifier>>>(x.t), ": ");
    Walk(std::get<ScalarIntExpr>(x.t));
  }
  void Unparse(const OmpEnterClause &x) {
    using Modifier = OmpEnterClause::Modifier;
    Walk(std::get<std::optional<std::list<Modifier>>>(x.t), ": ");
    Walk(std::get<OmpObjectList>(x.t));
  }
  void Unparse(const OmpFromClause &x) {
    using Modifier = OmpFromClause::Modifier;
    Walk(std::get<std::optional<std::list<Modifier>>>(x.t), ": ");
    Walk(std::get<OmpObjectList>(x.t));
  }
  void Unparse(const OmpIfClause &x) {
    using Modifier = OmpIfClause::Modifier;
    Walk(std::get<std::optional<std::list<Modifier>>>(x.t), ": ");
    Walk(std::get<ScalarLogicalExpr>(x.t));
  }
  void Unparse(const OmpStepSimpleModifier &x) { Walk(x.v); }
  void Unparse(const OmpStepComplexModifier &x) {
    Word("STEP(");
    Walk(x.v);
    Put(")");
  }
  void Unparse(const OmpLinearClause &x) {
    using Modifier = OmpLinearClause::Modifier;
    auto &modifiers{std::get<std::optional<std::list<Modifier>>>(x.t)};
    if (std::get<bool>(x.t)) { // PostModified
      Walk(std::get<OmpObjectList>(x.t));
      Walk(": ", modifiers);
    } else {
      // Unparse using pre-5.2 syntax.
      bool HasStepModifier{false}, HasLinearModifier{false};

      if (modifiers) {
        bool NeedComma{false};
        for (const Modifier &m : *modifiers) {
          // Print all linear modifiers in case we need to unparse an
          // incorrect tree.
          if (auto *lmod{std::get_if<parser::OmpLinearModifier>(&m.u)}) {
            if (NeedComma) {
              Put(",");
            }
            Walk(*lmod);
            HasLinearModifier = true;
            NeedComma = true;
          } else {
            // If not linear-modifier, then it has to be step modifier.
            HasStepModifier = true;
          }
        }
      }

      if (HasLinearModifier) {
        Put("(");
      }
      Walk(std::get<OmpObjectList>(x.t));
      if (HasLinearModifier) {
        Put(")");
      }

      if (HasStepModifier) {
        Put(": ");
        bool NeedComma{false};
        for (const Modifier &m : *modifiers) {
          if (!std::holds_alternative<parser::OmpLinearModifier>(m.u)) {
            if (NeedComma) {
              Put(",");
            }
            common::visit([&](auto &&s) { Walk(s); }, m.u);
            NeedComma = true;
          }
        }
      }
    }
  }
  void Unparse(const OmpReductionClause &x) {
    using Modifier = OmpReductionClause::Modifier;
    Walk(std::get<std::optional<std::list<Modifier>>>(x.t), ": ");
    Walk(std::get<OmpObjectList>(x.t));
  }
  void Unparse(const OmpDetachClause &x) { Walk(x.v); }
  void Unparse(const OmpInReductionClause &x) {
    using Modifier = OmpInReductionClause::Modifier;
    Walk(std::get<std::optional<std::list<Modifier>>>(x.t), ": ");
    Walk(std::get<OmpObjectList>(x.t));
  }
  void Unparse(const OmpTaskReductionClause &x) {
    using Modifier = OmpTaskReductionClause::Modifier;
    Walk(std::get<std::optional<std::list<Modifier>>>(x.t), ": ");
    Walk(std::get<OmpObjectList>(x.t));
  }
  void Unparse(const OmpAllocateClause &x) {
    using Modifier = OmpAllocateClause::Modifier;
    Walk(std::get<std::optional<std::list<Modifier>>>(x.t), ": ");
    Walk(std::get<OmpObjectList>(x.t));
  }
  void Unparse(const OmpAlignModifier &x) {
    Word("ALIGN(");
    Walk(x.v);
    Put(")");
  }
  void Unparse(const OmpAllocatorSimpleModifier &x) { Walk(x.v); }
  void Unparse(const OmpAllocatorComplexModifier &x) {
    Word("ALLOCATOR(");
    Walk(x.v);
    Put(")");
  }
  void Unparse(const OmpOrderClause &x) {
    using Modifier = OmpOrderClause::Modifier;
    Walk(std::get<std::optional<std::list<Modifier>>>(x.t), ":");
    Walk(std::get<OmpOrderClause::Ordering>(x.t));
  }
  void Unparse(const OmpGrainsizeClause &x) {
    using Modifier = OmpGrainsizeClause::Modifier;
    Walk(std::get<std::optional<std::list<Modifier>>>(x.t), ": ");
    Walk(std::get<ScalarIntExpr>(x.t));
  }
  void Unparse(const OmpNumTasksClause &x) {
    using Modifier = OmpNumTasksClause::Modifier;
    Walk(std::get<std::optional<std::list<Modifier>>>(x.t), ": ");
    Walk(std::get<ScalarIntExpr>(x.t));
  }
  void Unparse(const OmpDoacross::Sink &x) {
    Word("SINK: ");
    Walk(x.v.v);
  }
  void Unparse(const OmpDoacross::Source &) { Word("SOURCE"); }
  void Unparse(const OmpDependClause::TaskDep &x) {
    using Modifier = OmpDependClause::TaskDep::Modifier;
    Walk(std::get<std::optional<std::list<Modifier>>>(x.t), ": ");
    Walk(std::get<OmpObjectList>(x.t));
  }
  void Unparse(const OmpDefaultmapClause &x) {
    using Modifier = OmpDefaultmapClause::Modifier;
    Walk(std::get<OmpDefaultmapClause::ImplicitBehavior>(x.t));
    Walk(":", std::get<std::optional<std::list<Modifier>>>(x.t));
  }
  void Unparse(const OmpToClause &x) {
    using Modifier = OmpToClause::Modifier;
    Walk(std::get<std::optional<std::list<Modifier>>>(x.t), ": ");
    Walk(std::get<OmpObjectList>(x.t));
  }
  void Unparse(const OmpWhenClause &x) {
    using Modifier = OmpWhenClause::Modifier;
    using Directive = common::Indirection<OmpDirectiveSpecification>;
    Walk(std::get<std::optional<std::list<Modifier>>>(x.t), ": ");
    Walk(std::get<std::optional<Directive>>(x.t));
  }
#define GEN_FLANG_CLAUSE_UNPARSE
#include "llvm/Frontend/OpenMP/OMP.inc"
  void Unparse(const OmpLoopDirective &x) {
    switch (x.v) {
    case llvm::omp::Directive::OMPD_distribute:
      Word("DISTRIBUTE ");
      break;
    case llvm::omp::Directive::OMPD_distribute_parallel_do:
      Word("DISTRIBUTE PARALLEL DO ");
      break;
    case llvm::omp::Directive::OMPD_distribute_parallel_do_simd:
      Word("DISTRIBUTE PARALLEL DO SIMD ");
      break;
    case llvm::omp::Directive::OMPD_distribute_simd:
      Word("DISTRIBUTE SIMD ");
      break;
    case llvm::omp::Directive::OMPD_do:
      Word("DO ");
      break;
    case llvm::omp::Directive::OMPD_do_simd:
      Word("DO SIMD ");
      break;
    case llvm::omp::Directive::OMPD_loop:
      Word("LOOP ");
      break;
    case llvm::omp::Directive::OMPD_masked_taskloop_simd:
      Word("MASKED TASKLOOP SIMD");
      break;
    case llvm::omp::Directive::OMPD_masked_taskloop:
      Word("MASKED TASKLOOP");
      break;
    case llvm::omp::Directive::OMPD_master_taskloop_simd:
      Word("MASTER TASKLOOP SIMD");
      break;
    case llvm::omp::Directive::OMPD_master_taskloop:
      Word("MASTER TASKLOOP");
      break;
    case llvm::omp::Directive::OMPD_parallel_do:
      Word("PARALLEL DO ");
      break;
    case llvm::omp::Directive::OMPD_parallel_do_simd:
      Word("PARALLEL DO SIMD ");
      break;
    case llvm::omp::Directive::OMPD_parallel_masked_taskloop_simd:
      Word("PARALLEL MASKED TASKLOOP SIMD");
      break;
    case llvm::omp::Directive::OMPD_parallel_masked_taskloop:
      Word("PARALLEL MASKED TASKLOOP");
      break;
    case llvm::omp::Directive::OMPD_parallel_master_taskloop_simd:
      Word("PARALLEL MASTER TASKLOOP SIMD");
      break;
    case llvm::omp::Directive::OMPD_parallel_master_taskloop:
      Word("PARALLEL MASTER TASKLOOP");
      break;
    case llvm::omp::Directive::OMPD_simd:
      Word("SIMD ");
      break;
    case llvm::omp::Directive::OMPD_target_loop:
      Word("TARGET LOOP ");
      break;
    case llvm::omp::Directive::OMPD_target_parallel_do:
      Word("TARGET PARALLEL DO ");
      break;
    case llvm::omp::Directive::OMPD_target_parallel_do_simd:
      Word("TARGET PARALLEL DO SIMD ");
      break;
    case llvm::omp::Directive::OMPD_target_parallel_loop:
      Word("TARGET PARALLEL LOOP ");
      break;
    case llvm::omp::Directive::OMPD_target_teams_distribute:
      Word("TARGET TEAMS DISTRIBUTE ");
      break;
    case llvm::omp::Directive::OMPD_target_teams_distribute_parallel_do:
      Word("TARGET TEAMS DISTRIBUTE PARALLEL DO ");
      break;
    case llvm::omp::Directive::OMPD_target_teams_distribute_parallel_do_simd:
      Word("TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD ");
      break;
    case llvm::omp::Directive::OMPD_target_teams_distribute_simd:
      Word("TARGET TEAMS DISTRIBUTE SIMD ");
      break;
    case llvm::omp::Directive::OMPD_target_teams_loop:
      Word("TARGET TEAMS LOOP ");
      break;
    case llvm::omp::Directive::OMPD_target_simd:
      Word("TARGET SIMD ");
      break;
    case llvm::omp::Directive::OMPD_taskloop:
      Word("TASKLOOP ");
      break;
    case llvm::omp::Directive::OMPD_taskloop_simd:
      Word("TASKLOOP SIMD ");
      break;
    case llvm::omp::Directive::OMPD_teams_distribute:
      Word("TEAMS DISTRIBUTE ");
      break;
    case llvm::omp::Directive::OMPD_teams_distribute_parallel_do:
      Word("TEAMS DISTRIBUTE PARALLEL DO ");
      break;
    case llvm::omp::Directive::OMPD_teams_distribute_parallel_do_simd:
      Word("TEAMS DISTRIBUTE PARALLEL DO SIMD ");
      break;
    case llvm::omp::Directive::OMPD_teams_distribute_simd:
      Word("TEAMS DISTRIBUTE SIMD ");
      break;
    case llvm::omp::Directive::OMPD_tile:
      Word("TILE ");
      break;
    case llvm::omp::Directive::OMPD_unroll:
      Word("UNROLL ");
      break;
    default:
      break;
    }
  }
  void Unparse(const OmpObjectList &x) { Walk(x.v, ","); }

  void Unparse(const common::OmpMemoryOrderType &x) {
    Word(ToUpperCaseLetters(common::EnumToString(x)));
  }

  void Unparse(const OmpBeginDirective &x) {
    BeginOpenMP();
    Word("!$OMP ");
    Walk(static_cast<const OmpDirectiveSpecification &>(x));
    Put("\n");
    EndOpenMP();
  }

  void Unparse(const OmpEndDirective &x) {
    BeginOpenMP();
    Word("!$OMP END ");
    Walk(static_cast<const OmpDirectiveSpecification &>(x));
    Put("\n");
    EndOpenMP();
  }

  void Unparse(const OmpBlockConstruct &x) {
    Walk(std::get<OmpBeginDirective>(x.t));
    Walk(std::get<Block>(x.t), "");
    if (auto &end{std::get<std::optional<OmpEndDirective>>(x.t)}) {
      Walk(*end);
    } else {
      Put("\n");
    }
  }

  void Unparse(const OpenMPAtomicConstruct &x) { //
    Unparse(static_cast<const OmpBlockConstruct &>(x));
  }

  void Unparse(const OpenMPExecutableAllocate &x) {
    const auto &fields =
        std::get<std::optional<std::list<parser::OpenMPDeclarativeAllocate>>>(
            x.t);
    if (fields) {
      for (const auto &decl : *fields) {
        Walk(decl);
      }
    }
    BeginOpenMP();
    Word("!$OMP ALLOCATE");
    Walk(" (", std::get<std::optional<OmpObjectList>>(x.t), ")");
    Walk(std::get<OmpClauseList>(x.t));
    Put("\n");
    EndOpenMP();
    Walk(std::get<Statement<AllocateStmt>>(x.t));
  }
  void Unparse(const OpenMPDeclarativeAllocate &x) {
    BeginOpenMP();
    Word("!$OMP ALLOCATE");
    Put(" (");
    Walk(std::get<OmpObjectList>(x.t));
    Put(")");
    Walk(std::get<OmpClauseList>(x.t));
    Put("\n");
    EndOpenMP();
  }
  void Unparse(const OpenMPAllocatorsConstruct &x) {
    Unparse(static_cast<const OmpBlockConstruct &>(x));
  }
  void Unparse(const OpenMPAssumeConstruct &x) {
    Unparse(static_cast<const OmpBlockConstruct &>(x));
  }
  void Unparse(const OpenMPCriticalConstruct &x) {
    Unparse(static_cast<const OmpBlockConstruct &>(x));
  }
  void Unparse(const OmpDeclareTargetWithList &x) {
    Put("("), Walk(x.v), Put(")");
  }
  void Unparse(const OmpInitializerProc &x) {
    Walk(std::get<ProcedureDesignator>(x.t));
    Put("(");
    Walk(std::get<std::list<ActualArgSpec>>(x.t));
    Put(")");
  }
  void Unparse(const OmpInitializerClause &x) {
    // Don't let the visitor go to the normal AssignmentStmt Unparse function,
    // it adds an extra newline that we don't want.
    if (const auto *assignment{std::get_if<AssignmentStmt>(&x.u)}) {
      Walk(assignment->t, "=");
    } else {
      Walk(x.u);
    }
  }
  void Unparse(const OpenMPDeclareReductionConstruct &x) {
    BeginOpenMP();
    Word("!$OMP DECLARE REDUCTION ");
    Put("(");
    Walk(std::get<common::Indirection<OmpReductionSpecifier>>(x.t));
    Put(")");
    Walk(std::get<std::optional<OmpClauseList>>(x.t));
    Put("\n");
    EndOpenMP();
  }
  void Unparse(const OmpAppendArgsClause::OmpAppendOp &x) {
    Put("INTEROP(");
    Walk(x.v, ",");
    Put(")");
  }
  void Unparse(const OmpAppendArgsClause &x) { Walk(x.v, ","); }
  void Unparse(const OmpAdjustArgsClause &x) {
    Walk(std::get<OmpAdjustArgsClause::OmpAdjustOp>(x.t).v);
    Put(":");
    Walk(std::get<parser::OmpObjectList>(x.t));
  }
  void Unparse(const OmpDeclareVariantDirective &x) {
    BeginOpenMP();
    Word("!$OMP DECLARE VARIANT ");
    Put("(");
    Walk(std::get<std::optional<Name>>(x.t), ":");
    Walk(std::get<Name>(x.t));
    Put(")");
    Walk(std::get<OmpClauseList>(x.t));
    Put("\n");
    EndOpenMP();
  }
  void Unparse(const OpenMPInteropConstruct &x) {
    BeginOpenMP();
    Word("!$OMP INTEROP");
    using Flags = OmpDirectiveSpecification::Flags;
    if (std::get<Flags>(x.v.t) == Flags::DeprecatedSyntax) {
      Walk("(", std::get<std::optional<OmpArgumentList>>(x.v.t), ")");
      Walk(" ", std::get<std::optional<OmpClauseList>>(x.v.t));
    } else {
      Walk(" ", std::get<std::optional<OmpClauseList>>(x.v.t));
      Walk(" (", std::get<std::optional<OmpArgumentList>>(x.v.t), ")");
    }
    Put("\n");
    EndOpenMP();
  }

  void Unparse(const OpenMPDeclarativeAssumes &x) {
    BeginOpenMP();
    Word("!$OMP ASSUMES ");
    Walk(std::get<OmpClauseList>(x.t));
    Put("\n");
    EndOpenMP();
  }
  void Unparse(const OpenMPDeclareMapperConstruct &z) {
    BeginOpenMP();
    Word("!$OMP DECLARE MAPPER (");
    const auto &spec{std::get<OmpMapperSpecifier>(z.t)};
    const auto &mapperName{std::get<std::string>(spec.t)};
    if (mapperName.find(llvm::omp::OmpDefaultMapperName) == std::string::npos) {
      Walk(mapperName);
      Put(":");
    }
    Walk(std::get<TypeSpec>(spec.t));
    Put("::");
    Walk(std::get<Name>(spec.t));
    Put(")");

    Walk(std::get<OmpClauseList>(z.t));
    Put("\n");
    EndOpenMP();
  }
  void Unparse(const OpenMPDeclareSimdConstruct &y) {
    BeginOpenMP();
    Word("!$OMP DECLARE SIMD ");
    Walk("(", std::get<std::optional<Name>>(y.t), ")");
    Walk(std::get<OmpClauseList>(y.t));
    Put("\n");
    EndOpenMP();
  }
  void Unparse(const OpenMPDeclareTargetConstruct &x) {
    BeginOpenMP();
    Word("!$OMP DECLARE TARGET ");
    Walk(std::get<parser::OmpDeclareTargetSpecifier>(x.t));
    Put("\n");
    EndOpenMP();
  }
  void Unparse(const OpenMPDispatchConstruct &x) { //
    Unparse(static_cast<const OmpBlockConstruct &>(x));
  }
  void Unparse(const OpenMPGroupprivate &x) {
    BeginOpenMP();
    Word("!$OMP ");
    Walk(x.v);
    Put("\n");
    EndOpenMP();
  }
  void Unparse(const OpenMPRequiresConstruct &y) {
    BeginOpenMP();
    Word("!$OMP REQUIRES ");
    Walk(std::get<OmpClauseList>(y.t));
    Put("\n");
    EndOpenMP();
  }
  void Unparse(const OpenMPThreadprivate &x) {
    BeginOpenMP();
    Word("!$OMP THREADPRIVATE (");
    Walk(std::get<parser::OmpObjectList>(x.t));
    Put(")\n");
    EndOpenMP();
  }

  bool Pre(const OmpMessageClause &x) {
    Walk(x.v);
    return false;
  }
  void Unparse(const OmpErrorDirective &x) {
    Word("!$OMP ERROR ");
    Walk(x.t);
    Put("\n");
  }
  void Unparse(const OmpNothingDirective &x) {
    Word("!$OMP NOTHING");
    Put("\n");
  }
  void Unparse(const OmpSectionsDirective &x) {
    switch (x.v) {
    case llvm::omp::Directive::OMPD_sections:
      Word("SECTIONS ");
      break;
    case llvm::omp::Directive::OMPD_parallel_sections:
      Word("PARALLEL SECTIONS ");
      break;
    default:
      break;
    }
  }
  void Unparse(const OpenMPSectionConstruct &x) {
    if (auto &&dirSpec{
            std::get<std::optional<OmpDirectiveSpecification>>(x.t)}) {
      BeginOpenMP();
      Word("!$OMP ");
      Walk(*dirSpec);
      Put("\n");
      EndOpenMP();
    }
    Walk(std::get<Block>(x.t), "");
  }
  void Unparse(const OpenMPSectionsConstruct &x) {
    BeginOpenMP();
    Word("!$OMP ");
    Walk(std::get<OmpBeginSectionsDirective>(x.t));
    Put("\n");
    EndOpenMP();
    Walk(std::get<std::list<OpenMPConstruct>>(x.t), "");
    BeginOpenMP();
    Word("!$OMP END ");
    Walk(std::get<std::optional<OmpEndSectionsDirective>>(x.t));
    Put("\n");
    EndOpenMP();
  }
  // Clause unparsers are usually generated by tablegen in the form
  // CLAUSE(VALUE). Here we only want to print VALUE so a custom unparser is
  // needed.
  void Unparse(const OmpClause::CancellationConstructType &x) { Walk(x.v); }
  void Unparse(const OpenMPCancellationPointConstruct &x) {
    BeginOpenMP();
    Word("!$OMP ");
    Walk(x.v);
    Put("\n");
    EndOpenMP();
  }
  void Unparse(const OpenMPCancelConstruct &x) {
    BeginOpenMP();
    Word("!$OMP ");
    Walk(x.v);
    Put("\n");
    EndOpenMP();
  }
  void Unparse(const OmpFailClause &x) { Walk(x.v); }
  void Unparse(const OmpMetadirectiveDirective &x) {
    BeginOpenMP();
    Word("!$OMP METADIRECTIVE ");
    Walk(std::get<OmpClauseList>(x.t));
    Put("\n");
    EndOpenMP();
  }
  void Unparse(const OpenMPDepobjConstruct &x) {
    BeginOpenMP();
    Word("!$OMP ");
    Walk(x.v);
    Put("\n");
    EndOpenMP();
  }
  void Unparse(const OpenMPFlushConstruct &x) {
    BeginOpenMP();
    Word("!$OMP FLUSH");
    using Flags = OmpDirectiveSpecification::Flags;
    if (std::get<Flags>(x.v.t) == Flags::DeprecatedSyntax) {
      Walk("(", std::get<std::optional<OmpArgumentList>>(x.v.t), ")");
      Walk(" ", std::get<std::optional<OmpClauseList>>(x.v.t));
    } else {
      Walk(" ", std::get<std::optional<OmpClauseList>>(x.v.t));
      Walk(" (", std::get<std::optional<OmpArgumentList>>(x.v.t), ")");
    }
    Put("\n");
    EndOpenMP();
  }
  void Unparse(const OmpEndLoopDirective &x) {
    BeginOpenMP();
    Word("!$OMP END ");
    Walk(std::get<OmpLoopDirective>(x.t));
    Walk(std::get<OmpClauseList>(x.t));
    Put("\n");
    EndOpenMP();
  }
  void Unparse(const OmpClauseList &x, const char *sep = " ") {
    Walk(" ", x.v, sep);
  }
  void Unparse(const OpenMPSimpleStandaloneConstruct &x) {
    BeginOpenMP();
    Word("!$OMP ");
    Walk(x.v);
    Put("\n");
    EndOpenMP();
  }
  void Unparse(const OpenMPBlockConstruct &x) {
    Unparse(static_cast<const OmpBlockConstruct &>(x));
  }
  void Unparse(const OpenMPLoopConstruct &x) {
    BeginOpenMP();
    Word("!$OMP ");
    Walk(std::get<OmpBeginLoopDirective>(x.t));
    Put("\n");
    EndOpenMP();
    Walk(std::get<std::optional<std::variant<DoConstruct,
            common::Indirection<parser::OpenMPLoopConstruct>>>>(x.t));
    Walk(std::get<std::optional<OmpEndLoopDirective>>(x.t));
  }
  void Unparse(const BasedPointer &x) {
    Put('('), Walk(std::get<0>(x.t)), Put(","), Walk(std::get<1>(x.t));
    Walk("(", std::get<std::optional<ArraySpec>>(x.t), ")"), Put(')');
  }
  void Unparse(const BasedPointerStmt &x) { Walk("POINTER ", x.v, ","); }
  void Unparse(const CUDAAttributesStmt &x) {
    Word("ATTRIBUTES("), Walk(std::get<common::CUDADataAttr>(x.t));
    Word(") "), Walk(std::get<std::list<Name>>(x.t), ", ");
  }
  void Post(const StructureField &x) {
    if (const auto *def{std::get_if<Statement<DataComponentDefStmt>>(&x.u)}) {
      for (const auto &item :
          std::get<std::list<ComponentOrFill>>(def->statement.t)) {
        if (const auto *comp{std::get_if<ComponentDecl>(&item.u)}) {
          structureComponents_.insert(std::get<Name>(comp->t).source);
        }
      }
    }
  }
  void Unparse(const StructureStmt &x) {
    Word("STRUCTURE ");
    // The name, if present, includes the /slashes/
    Walk(std::get<std::optional<Name>>(x.t));
    Walk(" ", std::get<std::list<EntityDecl>>(x.t), ", ");
    Indent();
  }
  void Post(const Union::UnionStmt &) { Word("UNION"), Indent(); }
  void Post(const Union::EndUnionStmt &) { Outdent(), Word("END UNION"); }
  void Post(const Map::MapStmt &) { Word("MAP"), Indent(); }
  void Post(const Map::EndMapStmt &) { Outdent(), Word("END MAP"); }
  void Post(const StructureDef::EndStructureStmt &) {
    Outdent(), Word("END STRUCTURE");
  }
  void Unparse(const OldParameterStmt &x) {
    Word("PARAMETER "), Walk(x.v, ", ");
  }
  void Unparse(const ArithmeticIfStmt &x) {
    Word("IF ("), Walk(std::get<Expr>(x.t)), Put(") ");
    Walk(std::get<1>(x.t)), Put(", ");
    Walk(std::get<2>(x.t)), Put(", ");
    Walk(std::get<3>(x.t));
  }
  void Unparse(const AssignStmt &x) {
    Word("ASSIGN "), Walk(std::get<Label>(x.t));
    Word(" TO "), Walk(std::get<Name>(x.t));
  }
  void Unparse(const AssignedGotoStmt &x) {
    Word("GO TO "), Walk(std::get<Name>(x.t));
    Walk(", (", std::get<std::list<Label>>(x.t), ", ", ")");
  }
  void Unparse(const PauseStmt &x) { Word("PAUSE"), Walk(" ", x.v); }

#define WALK_NESTED_ENUM(CLASS, ENUM) \
  void Unparse(const CLASS::ENUM &x) { Word(CLASS::EnumToString(x)); }
  WALK_NESTED_ENUM(AccDataModifier, Modifier)
  WALK_NESTED_ENUM(AccessSpec, Kind) // R807
  WALK_NESTED_ENUM(common, TypeParamAttr) // R734
  WALK_NESTED_ENUM(common, CUDADataAttr) // CUDA
  WALK_NESTED_ENUM(common, CUDASubprogramAttrs) // CUDA
  WALK_NESTED_ENUM(IntentSpec, Intent) // R826
  WALK_NESTED_ENUM(ImplicitStmt, ImplicitNoneNameSpec) // R866
  WALK_NESTED_ENUM(ConnectSpec::CharExpr, Kind) // R1205
  WALK_NESTED_ENUM(IoControlSpec::CharExpr, Kind)
  WALK_NESTED_ENUM(InquireSpec::CharVar, Kind)
  WALK_NESTED_ENUM(InquireSpec::IntVar, Kind)
  WALK_NESTED_ENUM(InquireSpec::LogVar, Kind)
  WALK_NESTED_ENUM(ProcedureStmt, Kind) // R1506
  WALK_NESTED_ENUM(UseStmt, ModuleNature) // R1410
  WALK_NESTED_ENUM(OmpAdjustArgsClause::OmpAdjustOp, Value) // OMP adjustop
  WALK_NESTED_ENUM(OmpAtClause, ActionTime) // OMP at
  WALK_NESTED_ENUM(OmpAutomapModifier, Value) // OMP automap-modifier
  WALK_NESTED_ENUM(OmpBindClause, Binding) // OMP bind
  WALK_NESTED_ENUM(OmpProcBindClause, AffinityPolicy) // OMP proc_bind
  WALK_NESTED_ENUM(OmpDefaultClause, DataSharingAttribute) // OMP default
  WALK_NESTED_ENUM(OmpDefaultmapClause, ImplicitBehavior) // OMP defaultmap
  WALK_NESTED_ENUM(OmpVariableCategory, Value) // OMP variable-category
  WALK_NESTED_ENUM(OmpLastprivateModifier, Value) // OMP lastprivate-modifier
  WALK_NESTED_ENUM(OmpChunkModifier, Value) // OMP chunk-modifier
  WALK_NESTED_ENUM(OmpLinearModifier, Value) // OMP linear-modifier
  WALK_NESTED_ENUM(OmpOrderingModifier, Value) // OMP ordering-modifier
  WALK_NESTED_ENUM(OmpTaskDependenceType, Value) // OMP task-dependence-type
  WALK_NESTED_ENUM(OmpScheduleClause, Kind) // OMP schedule-kind
  WALK_NESTED_ENUM(OmpSeverityClause, Severity) // OMP severity
  WALK_NESTED_ENUM(OmpAccessGroup, Value)
  WALK_NESTED_ENUM(OmpDeviceModifier, Value) // OMP device modifier
  WALK_NESTED_ENUM(
      OmpDeviceTypeClause, DeviceTypeDescription) // OMP device_type
  WALK_NESTED_ENUM(OmpReductionModifier, Value) // OMP reduction-modifier
  WALK_NESTED_ENUM(OmpExpectation, Value) // OMP motion-expectation
  WALK_NESTED_ENUM(OmpInteropType, Value) // OMP InteropType
  WALK_NESTED_ENUM(OmpOrderClause, Ordering) // OMP ordering
  WALK_NESTED_ENUM(OmpOrderModifier, Value) // OMP order-modifier
  WALK_NESTED_ENUM(OmpPrescriptiveness, Value) // OMP prescriptiveness
  WALK_NESTED_ENUM(OmpMapType, Value) // OMP map-type
  WALK_NESTED_ENUM(OmpMapTypeModifier, Value) // OMP map-type-modifier
  WALK_NESTED_ENUM(OmpAlwaysModifier, Value)
  WALK_NESTED_ENUM(OmpCloseModifier, Value)
  WALK_NESTED_ENUM(OmpDeleteModifier, Value)
  WALK_NESTED_ENUM(OmpPresentModifier, Value)
  WALK_NESTED_ENUM(OmpRefModifier, Value)
  WALK_NESTED_ENUM(OmpSelfModifier, Value)
  WALK_NESTED_ENUM(OmpTraitSelectorName, Value)
  WALK_NESTED_ENUM(OmpTraitSetSelectorName, Value)
  WALK_NESTED_ENUM(OmpxHoldModifier, Value)

#undef WALK_NESTED_ENUM
  void Unparse(const ReductionOperator::Operator x) {
    switch (x) {
    case ReductionOperator::Operator::Plus:
      Word("+");
      break;
    case ReductionOperator::Operator::Multiply:
      Word("*");
      break;
    case ReductionOperator::Operator::And:
      Word(".AND.");
      break;
    case ReductionOperator::Operator::Or:
      Word(".OR.");
      break;
    case ReductionOperator::Operator::Eqv:
      Word(".EQV.");
      break;
    case ReductionOperator::Operator::Neqv:
      Word(".NEQV.");
      break;
    default:
      Word(ReductionOperator::EnumToString(x));
      break;
    }
  }

  void Unparse(const CUFKernelDoConstruct::StarOrExpr &x) {
    if (x.v) {
      Walk(*x.v);
    } else {
      Word("*");
    }
  }
  void Unparse(const CUFKernelDoConstruct::LaunchConfiguration &x) {
    Word(" <<<");
    const auto &grid{std::get<0>(x.t)};
    if (grid.empty()) {
      Word("*");
    } else if (grid.size() == 1) {
      Walk(grid.front());
    } else {
      Walk("(", grid, ",", ")");
    }
    Word(",");
    const auto &block{std::get<1>(x.t)};
    if (block.empty()) {
      Word("*");
    } else if (block.size() == 1) {
      Walk(block.front());
    } else {
      Walk("(", block, ",", ")");
    }
    if (const auto &stream{std::get<2>(x.t)}) {
      Word(",STREAM="), Walk(*stream);
    }
    Word(">>>");
  }
  void Unparse(const CUFKernelDoConstruct::Directive &x) {
    Word("!$CUF KERNEL DO");
    Walk(" (", std::get<std::optional<ScalarIntConstantExpr>>(x.t), ")");
    Walk(std::get<std::optional<CUFKernelDoConstruct::LaunchConfiguration>>(
        x.t));
    Walk(" ", std::get<std::list<CUFReduction>>(x.t), " ");
    Word("\n");
  }
  void Unparse(const CUFKernelDoConstruct &x) {
    Walk(std::get<CUFKernelDoConstruct::Directive>(x.t));
    Walk(std::get<std::optional<DoConstruct>>(x.t));
  }
  void Unparse(const CUFReduction &x) {
    Word("REDUCE(");
    Walk(std::get<CUFReduction::Operator>(x.t));
    Walk(":", std::get<std::list<Scalar<Variable>>>(x.t), ",", ")");
  }

  void Done() const { CHECK(indent_ == 0); }

private:
  void Put(char);
  void Put(const char *);
  void Put(const std::string &);
  void PutNormalized(const std::string &);
  void PutKeywordLetter(char);
  void Word(const char *);
  void Word(const std::string &);
  void Word(const std::string_view &);
  void Indent() { indent_ += indentationAmount_; }
  void Outdent() {
    CHECK(indent_ >= indentationAmount_);
    indent_ -= indentationAmount_;
  }
  void BeginOpenMP() { openmpDirective_ = true; }
  void EndOpenMP() { openmpDirective_ = false; }
  void BeginOpenACC() { openaccDirective_ = true; }
  void EndOpenACC() { openaccDirective_ = false; }

  // Call back to the traversal framework.
  template <typename T> void Walk(const T &x) {
    Fortran::parser::Walk(x, *this);
  }

  // Traverse a std::optional<> value.  Emit a prefix and/or a suffix string
  // only when it contains a value.
  template <typename A>
  void Walk(
      const char *prefix, const std::optional<A> &x, const char *suffix = "") {
    if (x) {
      Word(prefix), Walk(*x), Word(suffix);
    }
  }
  template <typename A>
  void Walk(const std::optional<A> &x, const char *suffix = "") {
    return Walk("", x, suffix);
  }

  // Traverse a std::list<>.  Separate the elements with an optional string.
  // Emit a prefix and/or a suffix string only when the list is not empty.
  template <typename A>
  void Walk(const char *prefix, const std::list<A> &list,
      const char *comma = ", ", const char *suffix = "") {
    if (!list.empty()) {
      const char *str{prefix};
      for (const auto &x : list) {
        Word(str), Walk(x);
        str = comma;
      }
      Word(suffix);
    }
  }
  template <typename A>
  void Walk(const std::list<A> &list, const char *comma = ", ",
      const char *suffix = "") {
    return Walk("", list, comma, suffix);
  }

  // Traverse a std::tuple<>, with an optional separator.
  template <std::size_t J = 0, typename T>
  void WalkTupleElements(const T &tuple, const char *separator) {
    if (J > 0 && J < std::tuple_size_v<T>) {
      Word(separator); // this usage dodges "unused parameter" warning
    }
    if constexpr (J < std::tuple_size_v<T>) {
      Walk(std::get<J>(tuple));
      WalkTupleElements<J + 1>(tuple, separator);
    }
  }
  template <typename... A>
  void Walk(const std::tuple<A...> &tuple, const char *separator = "") {
    WalkTupleElements(tuple, separator);
  }

  void EndSubprogram(const char *kind, const std::optional<Name> &name) {
    Outdent(), Word("END "), Word(kind), Walk(" ", name);
    structureComponents_.clear();
  }

  llvm::raw_ostream &out_;
  const common::LangOptions &langOpts_;
  int indent_{0};
  const int indentationAmount_{1};
  int column_{1};
  const int maxColumns_{80};
  std::set<CharBlock> structureComponents_;
  Encoding encoding_{Encoding::UTF_8};
  bool capitalizeKeywords_{true};
  bool openaccDirective_{false};
  bool openmpDirective_{false};
  bool backslashEscapes_{false};
  preStatementType *preStatement_{nullptr};
  AnalyzedObjectsAsFortran *asFortran_{nullptr};
};

void UnparseVisitor::Put(char ch) {
  int sav = indent_;
  if (openmpDirective_ || openaccDirective_) {
    indent_ = 0;
  }
  if (column_ <= 1) {
    if (ch == '\n') {
      return;
    }
    for (int j{0}; j < indent_; ++j) {
      out_ << ' ';
    }
    column_ = indent_ + 2;
  } else if (ch == '\n') {
    column_ = 1;
  } else if (++column_ >= maxColumns_) {
    out_ << "&\n";
    for (int j{0}; j < indent_; ++j) {
      out_ << ' ';
    }
    if (openmpDirective_) {
      out_ << "!$OMP&";
      column_ = 8;
    } else if (openaccDirective_) {
      out_ << "!$ACC&";
      column_ = 8;
    } else {
      out_ << '&';
      column_ = indent_ + 3;
    }
  }
  out_ << ch;
  if (openmpDirective_ || openaccDirective_) {
    indent_ = sav;
  }
}

void UnparseVisitor::Put(const char *str) {
  for (; *str != '\0'; ++str) {
    Put(*str);
  }
}

void UnparseVisitor::Put(const std::string &str) {
  for (char ch : str) {
    Put(ch);
  }
}

void UnparseVisitor::PutNormalized(const std::string &str) {
  auto decoded{DecodeString<std::string, Encoding::LATIN_1>(str, true)};
  std::string encoded{EncodeString<Encoding::LATIN_1>(decoded)};
  Put(QuoteCharacterLiteral(encoded, backslashEscapes_));
}

void UnparseVisitor::PutKeywordLetter(char ch) {
  if (capitalizeKeywords_) {
    Put(ToUpperCaseLetter(ch));
  } else {
    Put(ToLowerCaseLetter(ch));
  }
}

void UnparseVisitor::Word(const char *str) {
  for (; *str != '\0'; ++str) {
    PutKeywordLetter(*str);
  }
}

void UnparseVisitor::Word(const std::string &str) { Word(str.c_str()); }

void UnparseVisitor::Word(const std::string_view &str) {
  for (std::size_t j{0}; j < str.length(); ++j) {
    PutKeywordLetter(str[j]);
  }
}

template <typename A>
void Unparse(llvm::raw_ostream &out, const A &root,
    const common::LangOptions &langOpts, Encoding encoding,
    bool capitalizeKeywords, bool backslashEscapes,
    preStatementType *preStatement, AnalyzedObjectsAsFortran *asFortran) {
  UnparseVisitor visitor{out, langOpts, 1, encoding, capitalizeKeywords,
      backslashEscapes, preStatement, asFortran};
  Walk(root, visitor);
  visitor.Done();
}

template void Unparse<Program>(llvm::raw_ostream &, const Program &,
    const common::LangOptions &, Encoding, bool, bool, preStatementType *,
    AnalyzedObjectsAsFortran *);
template void Unparse<Expr>(llvm::raw_ostream &, const Expr &,
    const common::LangOptions &, Encoding, bool, bool, preStatementType *,
    AnalyzedObjectsAsFortran *);

template void Unparse<parser::OpenMPDeclareReductionConstruct>(
    llvm::raw_ostream &, const parser::OpenMPDeclareReductionConstruct &,
    const common::LangOptions &, Encoding, bool, bool, preStatementType *,
    AnalyzedObjectsAsFortran *);
template void Unparse<parser::OmpMetadirectiveDirective>(llvm::raw_ostream &,
    const parser::OmpMetadirectiveDirective &, const common::LangOptions &,
    Encoding, bool, bool, preStatementType *, AnalyzedObjectsAsFortran *);
} // namespace Fortran::parser
