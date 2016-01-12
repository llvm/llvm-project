// RUN: %check_clang_tidy %s modernize-use-override %t

#define ABSTRACT = 0

#define OVERRIDE override
#define VIRTUAL virtual
#define NOT_VIRTUAL
#define NOT_OVERRIDE

#define MUST_USE_RESULT __attribute__((warn_unused_result))
#define UNUSED __attribute__((unused))

struct MUST_USE_RESULT MustUseResultObject {};

struct Base {
  virtual ~Base() {}
  virtual void a();
  virtual void b();
  virtual void c();
  virtual void d();
  virtual void d2();
  virtual void e() = 0;
  virtual void f() = 0;
  virtual void g() = 0;

  virtual void j() const;
  virtual MustUseResultObject k();
  virtual bool l() MUST_USE_RESULT UNUSED;
  virtual bool n() MUST_USE_RESULT UNUSED;

  virtual void m();
  virtual void m2();
  virtual void o() __attribute__((unused));

  virtual void r() &;
  virtual void rr() &&;

  virtual void cv() const volatile;
  virtual void cv2() const volatile;

  virtual void ne() noexcept(false);
  virtual void t() throw();
};

struct SimpleCases : public Base {
public:
  virtual ~SimpleCases();
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: prefer using 'override' or (rarely) 'final' instead of 'virtual' [modernize-use-override]
  // CHECK-FIXES: {{^}}  ~SimpleCases() override;

  void a();
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: annotate this
  // CHECK-FIXES: {{^}}  void a() override;

  void b() override;
  // CHECK-MESSAGES-NOT: warning:
  // CHECK-FIXES: {{^}}  void b() override;

  virtual void c();
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer using
  // CHECK-FIXES: {{^}}  void c() override;

  virtual void d() override;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: 'virtual' is redundant since the function is already declared 'override'
  // CHECK-FIXES: {{^}}  void d() override;

  virtual void d2() final;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: 'virtual' is redundant since the function is already declared 'final'
  // CHECK-FIXES: {{^}}  void d2() final;

  virtual void e() = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer using
  // CHECK-FIXES: {{^}}  void e() override = 0;

  virtual void f()=0;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer using
  // CHECK-FIXES: {{^}}  void f()override =0;

  virtual void g() ABSTRACT;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer using
  // CHECK-FIXES: {{^}}  void g() override ABSTRACT;

  virtual void j() const;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer using
  // CHECK-FIXES: {{^}}  void j() const override;

  virtual MustUseResultObject k();  // Has an implicit attribute.
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: prefer using
  // CHECK-FIXES: {{^}}  MustUseResultObject k() override;

  virtual bool l() MUST_USE_RESULT UNUSED;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer using
  // CHECK-FIXES: {{^}}  bool l() override MUST_USE_RESULT UNUSED;

  virtual bool n() UNUSED MUST_USE_RESULT;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer using
  // CHECK-FIXES: {{^}}  bool n() override UNUSED MUST_USE_RESULT;

  void m() override final;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: 'override' is redundant since the function is already declared 'final'
  // CHECK-FIXES: {{^}}  void m() final;

  virtual void m2() override final;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: 'virtual' and 'override' are redundant since the function is already declared 'final'
  // CHECK-FIXES: {{^}}  void m2() final;

  virtual void o() __attribute__((unused));
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer using
  // CHECK-FIXES: {{^}}  void o() override __attribute__((unused));

  virtual void ne() noexcept(false);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer using
  // CHECK-FIXES: {{^}}  void ne() noexcept(false) override;

  virtual void t() throw();
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer using
  // CHECK-FIXES: {{^}}  void t() throw() override;
};

// CHECK-MESSAGES-NOT: warning:

void SimpleCases::c() {}
// CHECK-FIXES: {{^}}void SimpleCases::c() {}

SimpleCases::~SimpleCases() {}
// CHECK-FIXES: {{^}}SimpleCases::~SimpleCases() {}

struct DefaultedDestructor : public Base {
  DefaultedDestructor() {}
  virtual ~DefaultedDestructor() = default;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: prefer using
  // CHECK-FIXES: {{^}}  ~DefaultedDestructor() override = default;
};

struct FinalSpecified : public Base {
public:
  virtual ~FinalSpecified() final;
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: 'virtual' is redundant since the function is already declared 'final'
  // CHECK-FIXES: {{^}}  ~FinalSpecified() final;

  void b() final;
  // CHECK-MESSAGES-NOT: warning:
  // CHECK-FIXES: {{^}}  void b() final;

  virtual void d() final;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: 'virtual' is redundant
  // CHECK-FIXES: {{^}}  void d() final;

  virtual void e() final = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: 'virtual' is redundant
  // CHECK-FIXES: {{^}}  void e() final = 0;

  virtual void j() const final;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: 'virtual' is redundant
  // CHECK-FIXES: {{^}}  void j() const final;

  virtual bool l() final MUST_USE_RESULT UNUSED;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: 'virtual' is redundant
  // CHECK-FIXES: {{^}}  bool l() final MUST_USE_RESULT UNUSED;
};

struct InlineDefinitions : public Base {
public:
  virtual ~InlineDefinitions() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: prefer using
  // CHECK-FIXES: {{^}}  ~InlineDefinitions() override {}

  void a() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: annotate this
  // CHECK-FIXES: {{^}}  void a() override {}

  void b() override {}
  // CHECK-MESSAGES-NOT: warning:
  // CHECK-FIXES: {{^}}  void b() override {}

  virtual void c()
  {}
  // CHECK-MESSAGES: :[[@LINE-2]]:16: warning: prefer using
  // CHECK-FIXES: {{^}}  void c() override

  virtual void d() override {}
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: 'virtual' is redundant
  // CHECK-FIXES: {{^}}  void d() override {}

  virtual void j() const
  {}
  // CHECK-MESSAGES: :[[@LINE-2]]:16: warning: prefer using
  // CHECK-FIXES: {{^}}  void j() const override

  virtual MustUseResultObject k() {}  // Has an implicit attribute.
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: prefer using
  // CHECK-FIXES: {{^}}  MustUseResultObject k() override {}

  virtual bool l() MUST_USE_RESULT UNUSED {}
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer using
  // CHECK-FIXES: {{^}}  bool l() override MUST_USE_RESULT UNUSED {}

  virtual void r() &
  {}
  // CHECK-MESSAGES: :[[@LINE-2]]:16: warning: prefer using
  // CHECK-FIXES: {{^}}  void r() & override

  virtual void rr() &&
  {}
  // CHECK-MESSAGES: :[[@LINE-2]]:16: warning: prefer using
  // CHECK-FIXES: {{^}}  void rr() && override

  virtual void cv() const volatile
  {}
  // CHECK-MESSAGES: :[[@LINE-2]]:16: warning: prefer using
  // CHECK-FIXES: {{^}}  void cv() const volatile override

  virtual void cv2() const volatile // some comment
  {}
  // CHECK-MESSAGES: :[[@LINE-2]]:16: warning: prefer using
  // CHECK-FIXES: {{^}}  void cv2() const volatile override // some comment
};

struct Macros : public Base {
  // Tests for 'virtual' and 'override' being defined through macros. Basically
  // give up for now.
  NOT_VIRTUAL void a() NOT_OVERRIDE;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: annotate this
  // CHECK-FIXES: {{^}}  NOT_VIRTUAL void a() override NOT_OVERRIDE;

  VIRTUAL void b() NOT_OVERRIDE;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer using
  // CHECK-FIXES: {{^}}  VIRTUAL void b() override NOT_OVERRIDE;

  NOT_VIRTUAL void c() OVERRIDE;
  // CHECK-MESSAGES-NOT: warning:
  // CHECK-FIXES: {{^}}  NOT_VIRTUAL void c() OVERRIDE;

  VIRTUAL void d() OVERRIDE;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: 'virtual' is redundant
  // CHECK-FIXES: {{^}}  VIRTUAL void d() OVERRIDE;

#define FUNC(return_type, name) return_type name()
  FUNC(void, e);
  // CHECK-FIXES: {{^}}  FUNC(void, e);

#define F virtual void f();
  F
  // CHECK-FIXES: {{^}}  F

  VIRTUAL void g() OVERRIDE final;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: 'virtual' and 'override' are redundant
  // CHECK-FIXES: {{^}}  VIRTUAL void g() final;
};

// Tests for templates.
template <typename T> struct TemplateBase {
  virtual void f(T t);
};

template <typename T> struct DerivedFromTemplate : public TemplateBase<T> {
  virtual void f(T t);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer using
  // CHECK-FIXES: {{^}}  void f(T t) override;
};
void f() { DerivedFromTemplate<int>().f(2); }

template <class C>
struct UnusedMemberInstantiation : public C {
  virtual ~UnusedMemberInstantiation() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: prefer using
  // CHECK-FIXES: {{^}}  ~UnusedMemberInstantiation() override {}
};
struct IntantiateWithoutUse : public UnusedMemberInstantiation<Base> {};

struct Base2 {
  virtual ~Base2() {}
  virtual void a();
};

// The OverrideAttr isn't propagated to specializations in all cases. Make sure
// we don't add "override" a second time.
template <int I>
struct MembersOfSpecializations : public Base2 {
  void a() override;
  // CHECK-MESSAGES-NOT: warning:
  // CHECK-FIXES: {{^}}  void a() override;
};
template <> void MembersOfSpecializations<3>::a() {}
void ff() { MembersOfSpecializations<3>().a(); };
