#include "../clang-tidy/utils/DeclRefExprUtils.h"
#include "ClangTidyDiagnosticConsumer.h"
#include "ClangTidyTest.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

namespace clang {
namespace tidy {

namespace {
using namespace clang::ast_matchers;

template <int Indirections>
class ConstReferenceDeclRefExprsTransform : public ClangTidyCheck {
public:
  ConstReferenceDeclRefExprsTransform(StringRef CheckName,
                                      ClangTidyContext *Context)
      : ClangTidyCheck(CheckName, Context) {}

  void registerMatchers(MatchFinder *Finder) override {
    Finder->addMatcher(varDecl(hasName("target")).bind("var"), this);
  }

  void check(const MatchFinder::MatchResult &Result) override {
    const auto *D = Result.Nodes.getNodeAs<VarDecl>("var");
    using utils::decl_ref_expr::constReferenceDeclRefExprs;
    const auto const_decrefexprs = constReferenceDeclRefExprs(
        *D, *cast<FunctionDecl>(D->getDeclContext())->getBody(),
        *Result.Context, Indirections);

    for (const DeclRefExpr *const Expr : const_decrefexprs) {
      assert(Expr);
      diag(Expr->getBeginLoc(), "const usage")
          << FixItHint::CreateInsertion(Expr->getBeginLoc(), "/*const*/");
    }
  }
};
} // namespace

namespace test {

template <int Indirections> void RunTest(StringRef Snippet) {

  StringRef CommonCode = R"(
    struct ConstTag{};
    struct NonConstTag{};
    struct Tag1{};

    struct S {
      void constMethod() const;
      void nonConstMethod();

      static void staticMethod();

      void operator()(ConstTag) const;
      void operator()(NonConstTag);

      void operator[](int);
      void operator[](int) const;

      int& at(int);
      const int& at(int) const;
      const int& at(Tag1);

      int& weird_overload();
      const double& weird_overload() const;

      bool operator==(const S&) const;

      int int_member;
      // We consider a mutation of the `*ptr_member` to be a const use of
      // `*this`. This is consistent with the semantics of `const`-qualified
      // methods, which prevent modifying `ptr_member` but not `*ptr_member`.
      int* ptr_member;

    };

    struct Derived : public S {

    };

    void useVal(S);
    void useRef(S&);
    void usePtr(S*);
    void usePtrPtr(S**);
    void usePtrConstPtr(S* const*);
    void useConstRef(const S&);
    void useConstPtr(const S*);
    void useConstPtrRef(const S*&);
    void useConstPtrPtr(const S**);
    void useConstPtrConstRef(const S* const&);
    void useConstPtrConstPtr(const S* const*);

    void useInt(int);
    void useIntRef(int&);
    void useIntConstRef(const int&);
    void useIntPtr(int*);
    void useIntConstPtr(const int*);

    )";

  std::string Code = (CommonCode + Snippet).str();

  llvm::SmallVector<StringRef, 1> Parts;
  StringRef(Code).split(Parts, "/*const*/");

  EXPECT_EQ(Code,
            runCheckOnCode<ConstReferenceDeclRefExprsTransform<Indirections>>(
                join(Parts, "")));
}

TEST(ConstReferenceDeclRefExprsTest, ConstValueVar) {
  RunTest<0>(R"(
    void f(const S target) {
      useVal(/*const*/target);
      useConstRef(/*const*/target);
      useConstPtr(&/*const*/target);
      useConstPtrConstRef(&/*const*/target);
      /*const*/target.constMethod();
      /*const*/target.staticMethod();
      /*const*/target(ConstTag{});
      /*const*/target[42];
      useConstRef((/*const*/target));
      (/*const*/target).constMethod();
      /*const*/target.staticMethod();
      (void)(/*const*/target == /*const*/target);
      (void)/*const*/target;
      (void)&/*const*/target;
      (void)*&/*const*/target;
      /*const*/target;
      S copy1 = /*const*/target;
      S copy2(/*const*/target);
      /*const*/target.int_member;
      useInt(/*const*/target.int_member);
      useIntConstRef(/*const*/target.int_member);
      useIntPtr(/*const*/target.ptr_member);
      useIntConstPtr(&/*const*/target.int_member);

      const S& const_target_ref = /*const*/target;
      const S* const_target_ptr = &/*const*/target;
    }
)");
}

TEST(ConstReferenceDeclRefExprsTest, ConstRefVar) {
  RunTest<0>(R"(
    void f(const S& target) {
      useVal(/*const*/target);
      useConstRef(/*const*/target);
      useConstPtr(&/*const*/target);
      useConstPtrConstRef(&/*const*/target);
      /*const*/target.constMethod();
      /*const*/target.staticMethod();
      /*const*/target(ConstTag{});
      /*const*/target[42];
      useConstRef((/*const*/target));
      (/*const*/target).constMethod();
      (void)(/*const*/target == /*const*/target);
      (void)/*const*/target;
      (void)&/*const*/target;
      (void)*&/*const*/target;
      /*const*/target;
      S copy1 = /*const*/target;
      S copy2(/*const*/target);
      /*const*/target.int_member;
      useInt(/*const*/target.int_member);
      useIntConstRef(/*const*/target.int_member);
      useIntPtr(/*const*/target.ptr_member);
      useIntConstPtr(&/*const*/target.int_member);
      (void)/*const*/target.at(3);

      const S& const_target_ref = /*const*/target;
      const S* const_target_ptr = &/*const*/target;
      (void)/*const*/target.at(3);
    }
)");
}

TEST(ConstReferenceDeclRefExprsTest, DEBUGREMOVEME) {
  RunTest<0>(R"(
    void f(S target, const S& other) {
      S* target_ptr = &target;
    }
)");
}

TEST(ConstReferenceDeclRefExprsTest, ValueVar) {
  RunTest<0>(R"(
    void f(S target, const S& other) {
      useConstRef(/*const*/target);
      useVal(/*const*/target);
      useConstPtr(&/*const*/target);
      useConstPtrConstRef(&/*const*/target);
      /*const*/target.constMethod();
      /*const*/target.staticMethod();
      target.nonConstMethod();
      /*const*/target(ConstTag{});
      /*const*/target[42];
      /*const*/target(ConstTag{});
      target(NonConstTag{});
      useRef(target);
      usePtr(&target);
      useConstRef((/*const*/target));
      (/*const*/target).constMethod();
      (void)(/*const*/target == /*const*/target);
      (void)(/*const*/target == other);
      (void)/*const*/target;
      (void)&/*const*/target;
      (void)*&/*const*/target;
      /*const*/target;
      S copy1 = /*const*/target;
      S copy2(/*const*/target);
      /*const*/target.int_member;
      useInt(/*const*/target.int_member);
      useIntConstRef(/*const*/target.int_member);
      useIntPtr(/*const*/target.ptr_member);
      useIntConstPtr(&/*const*/target.int_member);

      const S& const_target_ref = /*const*/target;
      const S* const_target_ptr = &/*const*/target;
      S* target_ptr = &target;

      (void)/*const*/target.at(3);
      ++target.at(3);
      const int civ = /*const*/target.at(3);
      const int& cir = /*const*/target.at(3);
      int& ir = target.at(3);
      target.at(Tag1{});
      target.weird_overload();
    }
)");
}

TEST(ConstReferenceDeclRefExprsTest, RefVar) {
  RunTest<0>(R"(
    void f(S& target) {
      useVal(/*const*/target);
      usePtr(&target);
      useConstRef(/*const*/target);
      useConstPtr(&/*const*/target);
      useConstPtrConstRef(&/*const*/target);
      /*const*/target.constMethod();
      /*const*/target.staticMethod();
      target.nonConstMethod();
      /*const*/target(ConstTag{});
      /*const*/target[42];
      useConstRef((/*const*/target));
      (/*const*/target).constMethod();
      (void)(/*const*/target == /*const*/target);
      (void)/*const*/target;
      (void)&/*const*/target;
      (void)*&/*const*/target;
      /*const*/target;
      S copy1 = /*const*/target;
      S copy2(/*const*/target);
      /*const*/target.int_member;
      useInt(/*const*/target.int_member);
      useIntConstRef(/*const*/target.int_member);
      useIntPtr(/*const*/target.ptr_member);
      useIntConstPtr(&/*const*/target.int_member);

      (void)(&/*const*/target)->int_member;
      useIntRef((&target)->int_member);

      const S& const_target_ref = /*const*/target;
      const S* const_target_ptr = &/*const*/target;
      S* target_ptr = &target;

      (void)/*const*/target.at(3);
      ++target.at(3);
      const int civ = /*const*/target.at(3);
      const int& cir = /*const*/target.at(3);
      int& ir = target.at(3);
      target.at(Tag1{});
      target.weird_overload();
    }
)");
}

TEST(ConstReferenceDeclRefExprsTest, PtrVar) {
  RunTest<1>(R"(
    void f(S* target) {
      useVal(*/*const*/target);
      usePtr(target);
      useConstRef(*/*const*/target);
      useConstPtr(/*const*/target);
      useConstPtrConstRef(/*const*/target);
      usePtrConstPtr(&target);
      /*const*/target->constMethod();
      /*const*/target->staticMethod();
      target->nonConstMethod();
      (*/*const*/target)(ConstTag{});
      (*/*const*/target)[42];
      /*const*/target->operator[](42);
      useConstRef((*/*const*/target));
      (/*const*/target)->constMethod();
      (void)(*/*const*/target == */*const*/target);
      (void)*/*const*/target;
      (void)/*const*/target;
      /*const*/target;
      S copy1 = */*const*/target;
      S copy2(*/*const*/target);
      /*const*/target->int_member;
      useInt(/*const*/target->int_member);
      useIntConstRef(/*const*/target->int_member);
      useIntPtr(/*const*/target->ptr_member);
      useIntConstPtr(&/*const*/target->int_member);

      const S& const_target_ref = */*const*/target;
      const S* const_target_ptr = /*const*/target;
      S* target_ptr = target;  // FIXME: we could chect const usage of `target_ptr`

      (void)/*const*/target->at(3);
      ++target->at(3);
      const int civ = /*const*/target->at(3);
      const int& cir = /*const*/target->at(3);
      int& ir = target->at(3);
      target->at(Tag1{});
      target->weird_overload();
    }
)");
}

TEST(ConstReferenceDeclRefExprsTest, ConstPtrVar) {
  RunTest<1>(R"(
    void f(const S* target) {
      useVal(*/*const*/target);
      useConstRef(*/*const*/target);
      useConstPtr(/*const*/target);
      useConstPtrRef(/*const*/target);
      useConstPtrPtr(&/*const*/target);
      useConstPtrConstPtr(&/*const*/target);
      useConstPtrConstRef(/*const*/target);
      /*const*/target->constMethod();
      /*const*/target->staticMethod();
      (*/*const*/target)(ConstTag{});
      (*/*const*/target)[42];
      /*const*/target->operator[](42);
      (void)(*/*const*/target == */*const*/target);
      (void)/*const*/target;
      (void)*/*const*/target;
      /*const*/target;
      if(/*const*/target) {}
      S copy1 = */*const*/target;
      S copy2(*/*const*/target);
      /*const*/target->int_member;
      useInt(/*const*/target->int_member);
      useIntConstRef(/*const*/target->int_member);
      useIntPtr(/*const*/target->ptr_member);
      useIntConstPtr(&/*const*/target->int_member);

      const S& const_target_ref = */*const*/target;
      const S* const_target_ptr = /*const*/target;

      (void)/*const*/target->at(3);
      const int civ = /*const*/target->at(3);
      const int& cir = /*const*/target->at(3);
    }
)");
}

TEST(ConstReferenceDeclRefExprsTest, ConstPtrPtrVar) {
  RunTest<2>(R"(
    void f(const S** target) {
      useVal(**/*const*/target);
      useConstRef(**/*const*/target);
      useConstPtr(*/*const*/target);
      useConstPtrRef(*/*const*/target);
      useConstPtrPtr(/*const*/target);
      useConstPtrConstPtr(/*const*/target);
      useConstPtrConstRef(*/*const*/target);
      (void)/*const*/target;
      (void)*/*const*/target;
      (void)**/*const*/target;
      /*const*/target;
      if(/*const*/target) {}
      if(*/*const*/target) {}
      S copy1 = **/*const*/target;
      S copy2(**/*const*/target);
      (*/*const*/target)->int_member;
      useInt((*/*const*/target)->int_member);
      useIntConstRef((*/*const*/target)->int_member);
      useIntPtr((*/*const*/target)->ptr_member);
      useIntConstPtr(&(*/*const*/target)->int_member);

      const S& const_target_ref = **/*const*/target;
      const S* const_target_ptr = */*const*/target;
    }
)");
}

TEST(ConstReferenceDeclRefExprsTest, ConstPtrConstPtrVar) {
  RunTest<2>(R"(
    void f(const S* const* target) {
      useVal(**/*const*/target);
      useConstRef(**/*const*/target);
      useConstPtr(*/*const*/target);
      useConstPtrConstPtr(/*const*/target);
      useConstPtrConstRef(*/*const*/target);
      (void)/*const*/target;
      (void)*/*const*/target;
      (void)**/*const*/target;
      /*const*/target;
      if(/*const*/target) {}
      if(*/*const*/target) {}
      S copy1 = **/*const*/target;
      S copy2(**/*const*/target);
      (*/*const*/target)->int_member;
      useInt((*/*const*/target)->int_member);
      useIntConstRef((*/*const*/target)->int_member);
      useIntPtr((*/*const*/target)->ptr_member);
      useIntConstPtr(&(*/*const*/target)->int_member);

      const S& const_target_ref = **/*const*/target;
      const S* const_target_ptr = */*const*/target;
    }
)");
}

} // namespace test
} // namespace tidy
} // namespace clang
