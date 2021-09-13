// RUN: %check_clang_tidy %s cppcoreguidelines-virtual-class-destructor %t -- --fix-notes

// CHECK-MESSAGES: :[[@LINE+4]]:8: warning: destructor of 'PrivateVirtualBaseStruct' is private and prevents using the type [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+3]]:8: note: make it public and virtual
// CHECK-MESSAGES: :[[@LINE+2]]:8: note: make it protected
// As we have 2 conflicting fixes in notes, no fix is applied.
struct PrivateVirtualBaseStruct {
  virtual void f();

private:
  virtual ~PrivateVirtualBaseStruct() {}
};

struct PublicVirtualBaseStruct { // OK
  virtual void f();
  virtual ~PublicVirtualBaseStruct() {}
};

// CHECK-MESSAGES: :[[@LINE+2]]:8: warning: destructor of 'ProtectedVirtualBaseStruct' is protected and virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+1]]:8: note: make it protected and non-virtual
struct ProtectedVirtualBaseStruct {
  virtual void f();

protected:
  virtual ~ProtectedVirtualBaseStruct() {}
  // CHECK-FIXES: ~ProtectedVirtualBaseStruct() {}
};

// CHECK-MESSAGES: :[[@LINE+2]]:8: warning: destructor of 'ProtectedVirtualDefaultBaseStruct' is protected and virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+1]]:8: note: make it protected and non-virtual
struct ProtectedVirtualDefaultBaseStruct {
  virtual void f();

protected:
  virtual ~ProtectedVirtualDefaultBaseStruct() = default;
  // CHECK-FIXES: ~ProtectedVirtualDefaultBaseStruct() = default;
};

// CHECK-MESSAGES: :[[@LINE+4]]:8: warning: destructor of 'PrivateNonVirtualBaseStruct' is private and prevents using the type [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+3]]:8: note: make it public and virtual
// CHECK-MESSAGES: :[[@LINE+2]]:8: note: make it protected
// As we have 2 conflicting fixes in notes, no fix is applied.
struct PrivateNonVirtualBaseStruct {
  virtual void f();

private:
  ~PrivateNonVirtualBaseStruct() {}
};

// CHECK-MESSAGES: :[[@LINE+2]]:8: warning: destructor of 'PublicNonVirtualBaseStruct' is public and non-virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+1]]:8: note: make it public and virtual
struct PublicNonVirtualBaseStruct {
  virtual void f();
  ~PublicNonVirtualBaseStruct() {}
  // CHECK-FIXES: virtual ~PublicNonVirtualBaseStruct() {}
};

struct PublicNonVirtualNonBaseStruct { // OK according to C.35, since this struct does not have any virtual methods.
  void f();
  ~PublicNonVirtualNonBaseStruct() {}
};

// CHECK-MESSAGES: :[[@LINE+4]]:8: warning: destructor of 'PublicImplicitNonVirtualBaseStruct' is public and non-virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+3]]:8: note: make it public and virtual
// CHECK-FIXES: struct PublicImplicitNonVirtualBaseStruct {
// CHECK-FIXES-NEXT: virtual ~PublicImplicitNonVirtualBaseStruct() = default;
struct PublicImplicitNonVirtualBaseStruct {
  virtual void f();
};

// CHECK-MESSAGES: :[[@LINE+5]]:8: warning: destructor of 'PublicASImplicitNonVirtualBaseStruct' is public and non-virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+4]]:8: note: make it public and virtual
// CHECK-FIXES: struct PublicASImplicitNonVirtualBaseStruct {
// CHECK-FIXES-NEXT: virtual ~PublicASImplicitNonVirtualBaseStruct() = default;
// CHECK-FIXES-NEXT: private:
struct PublicASImplicitNonVirtualBaseStruct {
private:
  virtual void f();
};

struct ProtectedNonVirtualBaseStruct { // OK
  virtual void f();

protected:
  ~ProtectedNonVirtualBaseStruct() {}
};

// CHECK-MESSAGES: :[[@LINE+4]]:7: warning: destructor of 'PrivateVirtualBaseClass' is private and prevents using the type [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+3]]:7: note: make it public and virtual
// CHECK-MESSAGES: :[[@LINE+2]]:7: note: make it protected
// As we have 2 conflicting fixes in notes, no fix is applied.
class PrivateVirtualBaseClass {
  virtual void f();
  virtual ~PrivateVirtualBaseClass() {}
};

class PublicVirtualBaseClass { // OK
  virtual void f();

public:
  virtual ~PublicVirtualBaseClass() {}
};

// CHECK-MESSAGES: :[[@LINE+2]]:7: warning: destructor of 'ProtectedVirtualBaseClass' is protected and virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+1]]:7: note: make it protected and non-virtual
class ProtectedVirtualBaseClass {
  virtual void f();

protected:
  virtual ~ProtectedVirtualBaseClass() {}
  // CHECK-FIXES: ~ProtectedVirtualBaseClass() {}
};

// CHECK-MESSAGES: :[[@LINE+5]]:7: warning: destructor of 'PublicImplicitNonVirtualBaseClass' is public and non-virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+4]]:7: note: make it public and virtual
// CHECK-FIXES: public:
// CHECK-FIXES-NEXT: virtual ~PublicImplicitNonVirtualBaseClass() = default;
// CHECK-FIXES-NEXT: };
class PublicImplicitNonVirtualBaseClass {
  virtual void f();
};

// CHECK-MESSAGES: :[[@LINE+6]]:7: warning: destructor of 'PublicASImplicitNonVirtualBaseClass' is public and non-virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+5]]:7: note: make it public and virtual
// CHECK-FIXES: public:
// CHECK-FIXES-NEXT: virtual ~PublicASImplicitNonVirtualBaseClass() = default;
// CHECK-FIXES-NEXT: int foo = 42;
// CHECK-FIXES-NEXT: };
class PublicASImplicitNonVirtualBaseClass {
  virtual void f();

public:
  int foo = 42;
};

// CHECK-MESSAGES: :[[@LINE+2]]:7: warning: destructor of 'PublicNonVirtualBaseClass' is public and non-virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+1]]:7: note: make it public and virtual
class PublicNonVirtualBaseClass {
  virtual void f();

public:
  ~PublicNonVirtualBaseClass() {}
  // CHECK-FIXES: virtual ~PublicNonVirtualBaseClass() {}
};

class PublicNonVirtualNonBaseClass { // OK accoring to C.35, since this class does not have any virtual methods.
  void f();

public:
  ~PublicNonVirtualNonBaseClass() {}
};

class ProtectedNonVirtualClass { // OK
public:
  virtual void f();

protected:
  ~ProtectedNonVirtualClass() {}
};

// CHECK-MESSAGES: :[[@LINE+7]]:7: warning: destructor of 'OverridingDerivedClass' is public and non-virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+6]]:7: note: make it public and virtual
// CHECK-FIXES: class OverridingDerivedClass : ProtectedNonVirtualClass {
// CHECK-FIXES-NEXT: public:
// CHECK-FIXES-NEXT: virtual ~OverridingDerivedClass() = default;
// CHECK-FIXES-NEXT: void f() override;
// CHECK-FIXES-NEXT: };
class OverridingDerivedClass : ProtectedNonVirtualClass {
public:
  void f() override; // is implicitly virtual
};

// CHECK-MESSAGES: :[[@LINE+7]]:7: warning: destructor of 'NonOverridingDerivedClass' is public and non-virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+6]]:7: note: make it public and virtual
// CHECK-FIXES: class NonOverridingDerivedClass : ProtectedNonVirtualClass {
// CHECK-FIXES-NEXT: void m();
// CHECK-FIXES-NEXT: public:
// CHECK-FIXES-NEXT: virtual ~NonOverridingDerivedClass() = default;
// CHECK-FIXES-NEXT: };
class NonOverridingDerivedClass : ProtectedNonVirtualClass {
  void m();
};
// inherits virtual method

// CHECK-MESSAGES: :[[@LINE+6]]:8: warning: destructor of 'OverridingDerivedStruct' is public and non-virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+5]]:8: note: make it public and virtual
// CHECK-FIXES: struct OverridingDerivedStruct : ProtectedNonVirtualBaseStruct {
// CHECK-FIXES-NEXT: virtual ~OverridingDerivedStruct() = default;
// CHECK-FIXES-NEXT: void f() override;
// CHECK-FIXES-NEXT: };
struct OverridingDerivedStruct : ProtectedNonVirtualBaseStruct {
  void f() override; // is implicitly virtual
};

// CHECK-MESSAGES: :[[@LINE+6]]:8: warning: destructor of 'NonOverridingDerivedStruct' is public and non-virtual [cppcoreguidelines-virtual-class-destructor]
// CHECK-MESSAGES: :[[@LINE+5]]:8: note: make it public and virtual
// CHECK-FIXES: struct NonOverridingDerivedStruct : ProtectedNonVirtualBaseStruct {
// CHECK-FIXES-NEXT: virtual ~NonOverridingDerivedStruct() = default;
// CHECK-FIXES-NEXT: void m();
// CHECK-FIXES-NEXT: };
struct NonOverridingDerivedStruct : ProtectedNonVirtualBaseStruct {
  void m();
};
// inherits virtual method
