// RUN: %clang_cc1 -fsyntax-only -verify -Wthread-safety %s

#define LOCKABLE            __attribute__ ((lockable))
#define SCOPED_LOCKABLE     __attribute__ ((scoped_lockable))
#define GUARDED_BY(x)       __attribute__ ((guarded_by(x)))
#define GUARDED_VAR         __attribute__ ((guarded_var))
#define PT_GUARDED_BY(x)    __attribute__ ((pt_guarded_by(x)))
#define PT_GUARDED_VAR      __attribute__ ((pt_guarded_var))
#define ACQUIRED_AFTER(...) __attribute__ ((acquired_after(__VA_ARGS__)))
#define ACQUIRED_BEFORE(...) __attribute__ ((acquired_before(__VA_ARGS__)))
#define EXCLUSIVE_LOCK_FUNCTION(...)   __attribute__ ((exclusive_lock_function(__VA_ARGS__)))
#define SHARED_LOCK_FUNCTION(...)      __attribute__ ((shared_lock_function(__VA_ARGS__)))
#define EXCLUSIVE_TRYLOCK_FUNCTION(...) __attribute__ ((exclusive_trylock_function(__VA_ARGS__)))
#define SHARED_TRYLOCK_FUNCTION(...)    __attribute__ ((shared_trylock_function(__VA_ARGS__)))
#define UNLOCK_FUNCTION(...)            __attribute__ ((unlock_function(__VA_ARGS__)))
#define LOCK_RETURNED(x)    __attribute__ ((lock_returned(x)))
#define LOCKS_EXCLUDED(...) __attribute__ ((locks_excluded(__VA_ARGS__)))
#define EXCLUSIVE_LOCKS_REQUIRED(...) \
  __attribute__ ((exclusive_locks_required(__VA_ARGS__)))
#define SHARED_LOCKS_REQUIRED(...) \
  __attribute__ ((shared_locks_required(__VA_ARGS__)))
#define NO_THREAD_SAFETY_ANALYSIS  __attribute__ ((no_thread_safety_analysis))


class LOCKABLE Mutex {
  public:
  void Lock();
};

class UnlockableMu{
};

class MuWrapper {
  public:
  Mutex mu;
  Mutex getMu() {
    return mu;
  }
  Mutex * getMuPointer() {
    return &mu;
  }
};


class MuDoubleWrapper {
  public:
  MuWrapper* muWrapper;
  MuWrapper* getWrapper() {
    return muWrapper;
  }
};

Mutex mu1;
UnlockableMu umu;
Mutex mu2;
MuWrapper muWrapper;
MuDoubleWrapper muDoubleWrapper;
Mutex* muPointer;
Mutex** muDoublePointer = & muPointer;
Mutex& muRef = mu1;

//---------------------------------------//
// Scoping tests
//--------------------------------------//

class Foo {
  Mutex foomu;
  void needLock() EXCLUSIVE_LOCK_FUNCTION(foomu);
};

class Foo2 {
  void needLock() EXCLUSIVE_LOCK_FUNCTION(foomu);
  Mutex foomu;
};

class Bar {
 Mutex barmu;
 Mutex barmu2 ACQUIRED_AFTER(barmu);
};


//-----------------------------------------//
//   No Thread Safety Analysis (noanal)    //
//-----------------------------------------//

// FIXME: Right now we cannot parse attributes put on function definitions
// We would like to patch this at some point.

#if !__has_attribute(no_thread_safety_analysis)
#error "Should support no_thread_safety_analysis attribute"
#endif

void noanal_fun() NO_THREAD_SAFETY_ANALYSIS;

void noanal_fun_args() __attribute__((no_thread_safety_analysis(1))); // \
  // expected-error {{attribute takes no arguments}}

int noanal_testfn(int y) NO_THREAD_SAFETY_ANALYSIS;

int noanal_testfn(int y) {
  int x NO_THREAD_SAFETY_ANALYSIS = y; // \
    // expected-warning {{'no_thread_safety_analysis' attribute only applies to functions and methods}}
  return x;
};

int noanal_test_var NO_THREAD_SAFETY_ANALYSIS; // \
  // expected-warning {{'no_thread_safety_analysis' attribute only applies to functions and methods}}

class NoanalFoo {
 private:
  int test_field NO_THREAD_SAFETY_ANALYSIS; // \
    // expected-warning {{'no_thread_safety_analysis' attribute only applies to functions and methods}}
  void test_method() NO_THREAD_SAFETY_ANALYSIS;
};

class NO_THREAD_SAFETY_ANALYSIS NoanalTestClass { // \
  // expected-warning {{'no_thread_safety_analysis' attribute only applies to functions and methods}}
};

void noanal_fun_params(int lvar NO_THREAD_SAFETY_ANALYSIS); // \
  // expected-warning {{'no_thread_safety_analysis' attribute only applies to functions and methods}}


//-----------------------------------------//
//  Guarded Var Attribute (gv)
//-----------------------------------------//

#if !__has_attribute(guarded_var)
#error "Should support guarded_var attribute"
#endif

int gv_var_noargs GUARDED_VAR;

int gv_var_args __attribute__((guarded_var(1))); // \
  // expected-error {{attribute takes no arguments}}

class GVFoo {
 private:
  int gv_field_noargs GUARDED_VAR;
  int gv_field_args __attribute__((guarded_var(1))); // \
    // expected-error {{attribute takes no arguments}}
};

class GUARDED_VAR GV { // \
  // expected-warning {{'guarded_var' attribute only applies to fields and global variables}}
};

void gv_function() GUARDED_VAR; // \
  // expected-warning {{'guarded_var' attribute only applies to fields and global variables}}

void gv_function_params(int gv_lvar GUARDED_VAR); // \
  // expected-warning {{'guarded_var' attribute only applies to fields and global variables}}

int gv_testfn(int y){
  int x GUARDED_VAR = y; // \
    // expected-warning {{'guarded_var' attribute only applies to fields and global variables}}
  return x;
}

//-----------------------------------------//
//   Pt Guarded Var Attribute (pgv)
//-----------------------------------------//

//FIXME: add support for boost::scoped_ptr<int> fancyptr  and references

#if !__has_attribute(pt_guarded_var)
#error "Should support pt_guarded_var attribute"
#endif

int *pgv_pt_var_noargs PT_GUARDED_VAR;

int pgv_var_noargs PT_GUARDED_VAR; // \
    // expected-warning {{'pt_guarded_var' only applies to pointer types; type here is 'int'}}

class PGVFoo {
 private:
  int *pt_field_noargs PT_GUARDED_VAR;
  int field_noargs PT_GUARDED_VAR; // \
    // expected-warning {{'pt_guarded_var' only applies to pointer types; type here is 'int'}}
  int *gv_field_args __attribute__((pt_guarded_var(1))); // \
    // expected-error {{attribute takes no arguments}}
};

class PT_GUARDED_VAR PGV { // \
  // expected-warning {{'pt_guarded_var' attribute only applies to fields and global variables}}
};

int *pgv_var_args __attribute__((pt_guarded_var(1))); // \
  // expected-error {{attribute takes no arguments}}


void pgv_function() PT_GUARDED_VAR; // \
  // expected-warning {{'pt_guarded_var' attribute only applies to fields and global variables}}

void pgv_function_params(int *gv_lvar PT_GUARDED_VAR); // \
  // expected-warning {{'pt_guarded_var' attribute only applies to fields and global variables}}

void pgv_testfn(int y){
  int *x PT_GUARDED_VAR = new int(0); // \
    // expected-warning {{'pt_guarded_var' attribute only applies to fields and global variables}}
  delete x;
}

//-----------------------------------------//
//  Lockable Attribute (l)
//-----------------------------------------//

//FIXME: In future we may want to add support for structs, ObjC classes, etc.

#if !__has_attribute(lockable)
#error "Should support lockable attribute"
#endif

class LOCKABLE LTestClass {
};

class __attribute__((lockable (1))) LTestClass_args { // \
    // expected-error {{attribute takes no arguments}}
};

void l_test_function() LOCKABLE;  // \
  // expected-warning {{'lockable' attribute only applies to classes}}

int l_testfn(int y) {
  int x LOCKABLE = y; // \
    // expected-warning {{'lockable' attribute only applies to classes}}
  return x;
}

int l_test_var LOCKABLE; // \
  // expected-warning {{'lockable' attribute only applies to classes}}

class LFoo {
 private:
  int test_field LOCKABLE; // \
    // expected-warning {{'lockable' attribute only applies to classes}}
  void test_method() LOCKABLE; // \
    // expected-warning {{'lockable' attribute only applies to classes}}
};


void l_function_params(int lvar LOCKABLE); // \
  // expected-warning {{'lockable' attribute only applies to classes}}


//-----------------------------------------//
//  Scoped Lockable Attribute (sl)
//-----------------------------------------//

#if !__has_attribute(scoped_lockable)
#error "Should support scoped_lockable attribute"
#endif

class SCOPED_LOCKABLE SLTestClass {
};

class __attribute__((scoped_lockable (1))) SLTestClass_args { // \
  // expected-error {{attribute takes no arguments}}
};

void sl_test_function() SCOPED_LOCKABLE;  // \
  // expected-warning {{'scoped_lockable' attribute only applies to classes}}

int sl_testfn(int y) {
  int x SCOPED_LOCKABLE = y; // \
    // expected-warning {{'scoped_lockable' attribute only applies to classes}}
  return x;
}

int sl_test_var SCOPED_LOCKABLE; // \
  // expected-warning {{'scoped_lockable' attribute only applies to classes}}

class SLFoo {
 private:
  int test_field SCOPED_LOCKABLE; // \
    // expected-warning {{'scoped_lockable' attribute only applies to classes}}
  void test_method() SCOPED_LOCKABLE; // \
    // expected-warning {{'scoped_lockable' attribute only applies to classes}}
};


void sl_function_params(int lvar SCOPED_LOCKABLE); // \
  // expected-warning {{'scoped_lockable' attribute only applies to classes}}


//-----------------------------------------//
//  Guarded By Attribute (gb)
//-----------------------------------------//

// FIXME: Eventually, would we like this attribute to take more than 1 arg?

#if !__has_attribute(guarded_by)
#error "Should support guarded_by attribute"
#endif

//1. Check applied to the right types & argument number

int gb_var_arg GUARDED_BY(mu1);

int gb_var_args __attribute__((guarded_by(mu1, mu2))); // \
  // expected-error {{attribute takes one argument}}

int gb_var_noargs __attribute__((guarded_by)); // \
  // expected-error {{attribute takes one argument}}

class GBFoo {
 private:
  int gb_field_noargs __attribute__((guarded_by)); // \
    // expected-error {{attribute takes one argument}}
  int gb_field_args GUARDED_BY(mu1);
};

class GUARDED_BY(mu1) GB { // \
  // expected-warning {{'guarded_by' attribute only applies to fields and global variables}}
};

void gb_function() GUARDED_BY(mu1); // \
  // expected-warning {{'guarded_by' attribute only applies to fields and global variables}}

void gb_function_params(int gv_lvar GUARDED_BY(mu1)); // \
  // expected-warning {{'guarded_by' attribute only applies to fields and global variables}}

int gb_testfn(int y){
  int x GUARDED_BY(mu1) = y; // \
    // expected-warning {{'guarded_by' attribute only applies to fields and global variables}}
  return x;
}

//2. Check argument parsing.

// legal attribute arguments
int gb_var_arg_1 GUARDED_BY(muWrapper.mu);
int gb_var_arg_2 GUARDED_BY(muDoubleWrapper.muWrapper->mu);
int gb_var_arg_3 GUARDED_BY(muWrapper.getMu());
int gb_var_arg_4 GUARDED_BY(*muWrapper.getMuPointer());
int gb_var_arg_5 GUARDED_BY(&mu1);
int gb_var_arg_6 GUARDED_BY(muRef);
int gb_var_arg_7 GUARDED_BY(muDoubleWrapper.getWrapper()->getMu());
int gb_var_arg_8 GUARDED_BY(muPointer);


// illegal attribute arguments
int gb_var_arg_bad_1 GUARDED_BY(1); // \
  // expected-warning {{'guarded_by' attribute requires arguments that are class type or point to class type; type here is 'int'}}
int gb_var_arg_bad_2 GUARDED_BY("mu"); // \
  // expected-warning {{ignoring 'guarded_by' attribute because its argument is invalid}}
int gb_var_arg_bad_3 GUARDED_BY(muDoublePointer); // \
  // expected-warning {{'guarded_by' attribute requires arguments that are class type or point to class type; type here is 'class Mutex **'}}
int gb_var_arg_bad_4 GUARDED_BY(umu); // \
  // expected-warning {{'guarded_by' attribute requires arguments whose type is annotated with 'lockable' attribute; type here is 'class UnlockableMu'}}

//3.
// Thread Safety analysis tests


//-----------------------------------------//
//  Pt Guarded By Attribute (pgb)
//-----------------------------------------//

#if !__has_attribute(pt_guarded_by)
#error "Should support pt_guarded_by attribute"
#endif

//1. Check applied to the right types & argument number

int *pgb_var_noargs __attribute__((pt_guarded_by)); // \
  // expected-error {{attribute takes one argument}}

int *pgb_ptr_var_arg PT_GUARDED_BY(mu1);

int *pgb_ptr_var_args __attribute__((pt_guarded_by(mu1, mu2))); // \
  // expected-error {{attribute takes one argument}}

int pgb_var_args PT_GUARDED_BY(mu1); // \
  // expected-warning {{'pt_guarded_by' only applies to pointer types; type here is 'int'}}

class PGBFoo {
 private:
  int *pgb_field_noargs __attribute__((pt_guarded_by)); // \
    // expected-error {{attribute takes one argument}}
  int *pgb_field_args PT_GUARDED_BY(mu1);
};

class PT_GUARDED_BY(mu1) PGB { // \
  // expected-warning {{'pt_guarded_by' attribute only applies to fields and global variables}}
};

void pgb_function() PT_GUARDED_BY(mu1); // \
  // expected-warning {{'pt_guarded_by' attribute only applies to fields and global variables}}

void pgb_function_params(int gv_lvar PT_GUARDED_BY(mu1)); // \
  // expected-warning {{'pt_guarded_by' attribute only applies to fields and global variables}}

void pgb_testfn(int y){
  int *x PT_GUARDED_BY(mu1) = new int(0); // \
    // expected-warning {{'pt_guarded_by' attribute only applies to fields and global variables}}
  delete x;
}

//2. Check argument parsing.

// legal attribute arguments
int * pgb_var_arg_1 PT_GUARDED_BY(muWrapper.mu);
int * pgb_var_arg_2 PT_GUARDED_BY(muDoubleWrapper.muWrapper->mu);
int * pgb_var_arg_3 PT_GUARDED_BY(muWrapper.getMu());
int * pgb_var_arg_4 PT_GUARDED_BY(*muWrapper.getMuPointer());
int * pgb_var_arg_5 PT_GUARDED_BY(&mu1);
int * pgb_var_arg_6 PT_GUARDED_BY(muRef);
int * pgb_var_arg_7 PT_GUARDED_BY(muDoubleWrapper.getWrapper()->getMu());
int * pgb_var_arg_8 PT_GUARDED_BY(muPointer);


// illegal attribute arguments
int * pgb_var_arg_bad_1 PT_GUARDED_BY(1); // \
  // expected-warning {{'pt_guarded_by' attribute requires arguments that are class type or point to class type}}
int * pgb_var_arg_bad_2 PT_GUARDED_BY("mu"); // \
  // expected-warning {{ignoring 'pt_guarded_by' attribute because its argument is invalid}}
int * pgb_var_arg_bad_3 PT_GUARDED_BY(muDoublePointer); // \
  // expected-warning {{'pt_guarded_by' attribute requires arguments that are class type or point to class type}}
int * pgb_var_arg_bad_4 PT_GUARDED_BY(umu); // \
  // expected-warning {{'pt_guarded_by' attribute requires arguments whose type is annotated with 'lockable' attribute}}


//-----------------------------------------//
//  Acquired After (aa)
//-----------------------------------------//

// FIXME: Would we like this attribute to take more than 1 arg?

#if !__has_attribute(acquired_after)
#error "Should support acquired_after attribute"
#endif

Mutex mu_aa ACQUIRED_AFTER(mu1);

Mutex aa_var_noargs __attribute__((acquired_after)); // \
  // expected-error {{attribute takes at least 1 argument}}

class AAFoo {
 private:
  Mutex aa_field_noargs __attribute__((acquired_after)); // \
    // expected-error {{attribute takes at least 1 argument}}
  Mutex aa_field_args ACQUIRED_AFTER(mu1);
};

class ACQUIRED_AFTER(mu1) AA { // \
  // expected-warning {{'acquired_after' attribute only applies to fields and global variables}}
};

void aa_function() ACQUIRED_AFTER(mu1); // \
  // expected-warning {{'acquired_after' attribute only applies to fields and global variables}}

void aa_function_params(int gv_lvar ACQUIRED_AFTER(mu1)); // \
  // expected-warning {{'acquired_after' attribute only applies to fields and global variables}}

void aa_testfn(int y){
  Mutex x ACQUIRED_AFTER(mu1) = Mutex(); // \
    // expected-warning {{'acquired_after' attribute only applies to fields and global variables}}
}

//Check argument parsing.

// legal attribute arguments
Mutex aa_var_arg_1 ACQUIRED_AFTER(muWrapper.mu);
Mutex aa_var_arg_2 ACQUIRED_AFTER(muDoubleWrapper.muWrapper->mu);
Mutex aa_var_arg_3 ACQUIRED_AFTER(muWrapper.getMu());
Mutex aa_var_arg_4 ACQUIRED_AFTER(*muWrapper.getMuPointer());
Mutex aa_var_arg_5 ACQUIRED_AFTER(&mu1);
Mutex aa_var_arg_6 ACQUIRED_AFTER(muRef);
Mutex aa_var_arg_7 ACQUIRED_AFTER(muDoubleWrapper.getWrapper()->getMu());
Mutex aa_var_arg_8 ACQUIRED_AFTER(muPointer);


// illegal attribute arguments
Mutex aa_var_arg_bad_1 ACQUIRED_AFTER(1); // \
  // expected-warning {{'acquired_after' attribute requires arguments that are class type or point to class type}}
Mutex aa_var_arg_bad_2 ACQUIRED_AFTER("mu"); // \
  // expected-warning {{ignoring 'acquired_after' attribute because its argument is invalid}}
Mutex aa_var_arg_bad_3 ACQUIRED_AFTER(muDoublePointer); // \
  // expected-warning {{'acquired_after' attribute requires arguments that are class type or point to class type}}
Mutex aa_var_arg_bad_4 ACQUIRED_AFTER(umu); // \
  // expected-warning {{'acquired_after' attribute requires arguments whose type is annotated with 'lockable' attribute}}
UnlockableMu aa_var_arg_bad_5 ACQUIRED_AFTER(mu_aa); // \
  // expected-warning {{'acquired_after' attribute can only be applied in a context annotated with 'lockable' attribute}}

//-----------------------------------------//
//  Acquired Before (ab)
//-----------------------------------------//

#if !__has_attribute(acquired_before)
#error "Should support acquired_before attribute"
#endif

Mutex mu_ab ACQUIRED_BEFORE(mu1);

Mutex ab_var_noargs __attribute__((acquired_before)); // \
  // expected-error {{attribute takes at least 1 argument}}

class ABFoo {
 private:
  Mutex ab_field_noargs __attribute__((acquired_before)); // \
    // expected-error {{attribute takes at least 1 argument}}
  Mutex ab_field_args ACQUIRED_BEFORE(mu1);
};

class ACQUIRED_BEFORE(mu1) AB { // \
  // expected-warning {{'acquired_before' attribute only applies to fields and global variables}}
};

void ab_function() ACQUIRED_BEFORE(mu1); // \
  // expected-warning {{'acquired_before' attribute only applies to fields and global variables}}

void ab_function_params(int gv_lvar ACQUIRED_BEFORE(mu1)); // \
  // expected-warning {{'acquired_before' attribute only applies to fields and global variables}}

void ab_testfn(int y){
  Mutex x ACQUIRED_BEFORE(mu1) = Mutex(); // \
    // expected-warning {{'acquired_before' attribute only applies to fields and global variables}}
}

// Note: illegal int ab_int ACQUIRED_BEFORE(mu1) will
// be taken care of by warnings that ab__int is not lockable.

//Check argument parsing.

// legal attribute arguments
Mutex ab_var_arg_1 ACQUIRED_BEFORE(muWrapper.mu);
Mutex ab_var_arg_2 ACQUIRED_BEFORE(muDoubleWrapper.muWrapper->mu);
Mutex ab_var_arg_3 ACQUIRED_BEFORE(muWrapper.getMu());
Mutex ab_var_arg_4 ACQUIRED_BEFORE(*muWrapper.getMuPointer());
Mutex ab_var_arg_5 ACQUIRED_BEFORE(&mu1);
Mutex ab_var_arg_6 ACQUIRED_BEFORE(muRef);
Mutex ab_var_arg_7 ACQUIRED_BEFORE(muDoubleWrapper.getWrapper()->getMu());
Mutex ab_var_arg_8 ACQUIRED_BEFORE(muPointer);


// illegal attribute arguments
Mutex ab_var_arg_bad_1 ACQUIRED_BEFORE(1); // \
  // expected-warning {{'acquired_before' attribute requires arguments that are class type or point to class type}}
Mutex ab_var_arg_bad_2 ACQUIRED_BEFORE("mu"); // \
  // expected-warning {{ignoring 'acquired_before' attribute because its argument is invalid}}
Mutex ab_var_arg_bad_3 ACQUIRED_BEFORE(muDoublePointer); // \
  // expected-warning {{'acquired_before' attribute requires arguments that are class type or point to class type}}
Mutex ab_var_arg_bad_4 ACQUIRED_BEFORE(umu); // \
  // expected-warning {{'acquired_before' attribute requires arguments whose type is annotated with 'lockable' attribute}}
UnlockableMu ab_var_arg_bad_5 ACQUIRED_BEFORE(mu_ab); // \
  // expected-warning {{'acquired_before' attribute can only be applied in a context annotated with 'lockable' attribute}}


//-----------------------------------------//
//  Exclusive Lock Function (elf)
//-----------------------------------------//

#if !__has_attribute(exclusive_lock_function)
#error "Should support exclusive_lock_function attribute"
#endif

// takes zero or more arguments, all locks (vars/fields)

void elf_function() EXCLUSIVE_LOCK_FUNCTION();

void elf_function_args() EXCLUSIVE_LOCK_FUNCTION(mu1, mu2);

int elf_testfn(int y) EXCLUSIVE_LOCK_FUNCTION();

int elf_testfn(int y) {
  int x EXCLUSIVE_LOCK_FUNCTION() = y; // \
    // expected-warning {{'exclusive_lock_function' attribute only applies to functions and methods}}
  return x;
};

int elf_test_var EXCLUSIVE_LOCK_FUNCTION(); // \
  // expected-warning {{'exclusive_lock_function' attribute only applies to functions and methods}}

class ElfFoo {
 private:
  int test_field EXCLUSIVE_LOCK_FUNCTION(); // \
    // expected-warning {{'exclusive_lock_function' attribute only applies to functions and methods}}
  void test_method() EXCLUSIVE_LOCK_FUNCTION();
};

class EXCLUSIVE_LOCK_FUNCTION() ElfTestClass { // \
  // expected-warning {{'exclusive_lock_function' attribute only applies to functions and methods}}
};

void elf_fun_params(int lvar EXCLUSIVE_LOCK_FUNCTION()); // \
  // expected-warning {{'exclusive_lock_function' attribute only applies to functions and methods}}

// Check argument parsing.

// legal attribute arguments
int elf_function_1() EXCLUSIVE_LOCK_FUNCTION(muWrapper.mu);
int elf_function_2() EXCLUSIVE_LOCK_FUNCTION(muDoubleWrapper.muWrapper->mu);
int elf_function_3() EXCLUSIVE_LOCK_FUNCTION(muWrapper.getMu());
int elf_function_4() EXCLUSIVE_LOCK_FUNCTION(*muWrapper.getMuPointer());
int elf_function_5() EXCLUSIVE_LOCK_FUNCTION(&mu1);
int elf_function_6() EXCLUSIVE_LOCK_FUNCTION(muRef);
int elf_function_7() EXCLUSIVE_LOCK_FUNCTION(muDoubleWrapper.getWrapper()->getMu());
int elf_function_8() EXCLUSIVE_LOCK_FUNCTION(muPointer);
int elf_function_9(Mutex x) EXCLUSIVE_LOCK_FUNCTION(1);
int elf_function_9(Mutex x, Mutex y) EXCLUSIVE_LOCK_FUNCTION(1,2);


// illegal attribute arguments
int elf_function_bad_2() EXCLUSIVE_LOCK_FUNCTION("mu"); // \
  // expected-warning {{ignoring 'exclusive_lock_function' attribute because its argument is invalid}}
int elf_function_bad_3() EXCLUSIVE_LOCK_FUNCTION(muDoublePointer); // \
  // expected-warning {{'exclusive_lock_function' attribute requires arguments that are class type or point to class type}}
int elf_function_bad_4() EXCLUSIVE_LOCK_FUNCTION(umu); // \
  // expected-warning {{'exclusive_lock_function' attribute requires arguments whose type is annotated with 'lockable' attribute}}

int elf_function_bad_1() EXCLUSIVE_LOCK_FUNCTION(1); // \
  // expected-error {{'exclusive_lock_function' attribute parameter 1 is out of bounds: no parameters to index into}}
int elf_function_bad_5(Mutex x) EXCLUSIVE_LOCK_FUNCTION(0); // \
  // expected-error {{'exclusive_lock_function' attribute parameter 1 is out of bounds: can only be 1, since there is one parameter}}
int elf_function_bad_6(Mutex x, Mutex y) EXCLUSIVE_LOCK_FUNCTION(0); // \
  // expected-error {{'exclusive_lock_function' attribute parameter 1 is out of bounds: must be between 1 and 2}}
int elf_function_bad_7() EXCLUSIVE_LOCK_FUNCTION(0); // \
  // expected-error {{'exclusive_lock_function' attribute parameter 1 is out of bounds: no parameters to index into}}


//-----------------------------------------//
//  Shared Lock Function (slf)
//-----------------------------------------//

#if !__has_attribute(shared_lock_function)
#error "Should support shared_lock_function attribute"
#endif

// takes zero or more arguments, all locks (vars/fields)

void slf_function() SHARED_LOCK_FUNCTION();

void slf_function_args() SHARED_LOCK_FUNCTION(mu1, mu2);

int slf_testfn(int y) SHARED_LOCK_FUNCTION();

int slf_testfn(int y) {
  int x SHARED_LOCK_FUNCTION() = y; // \
    // expected-warning {{'shared_lock_function' attribute only applies to functions and methods}}
  return x;
};

int slf_test_var SHARED_LOCK_FUNCTION(); // \
  // expected-warning {{'shared_lock_function' attribute only applies to functions and methods}}

void slf_fun_params(int lvar SHARED_LOCK_FUNCTION()); // \
  // expected-warning {{'shared_lock_function' attribute only applies to functions and methods}}

class SlfFoo {
 private:
  int test_field SHARED_LOCK_FUNCTION(); // \
    // expected-warning {{'shared_lock_function' attribute only applies to functions and methods}}
  void test_method() SHARED_LOCK_FUNCTION();
};

class SHARED_LOCK_FUNCTION() SlfTestClass { // \
  // expected-warning {{'shared_lock_function' attribute only applies to functions and methods}}
};

// Check argument parsing.

// legal attribute arguments
int slf_function_1() SHARED_LOCK_FUNCTION(muWrapper.mu);
int slf_function_2() SHARED_LOCK_FUNCTION(muDoubleWrapper.muWrapper->mu);
int slf_function_3() SHARED_LOCK_FUNCTION(muWrapper.getMu());
int slf_function_4() SHARED_LOCK_FUNCTION(*muWrapper.getMuPointer());
int slf_function_5() SHARED_LOCK_FUNCTION(&mu1);
int slf_function_6() SHARED_LOCK_FUNCTION(muRef);
int slf_function_7() SHARED_LOCK_FUNCTION(muDoubleWrapper.getWrapper()->getMu());
int slf_function_8() SHARED_LOCK_FUNCTION(muPointer);
int slf_function_9(Mutex x) SHARED_LOCK_FUNCTION(1);
int slf_function_9(Mutex x, Mutex y) SHARED_LOCK_FUNCTION(1,2);


// illegal attribute arguments
int slf_function_bad_2() SHARED_LOCK_FUNCTION("mu"); // \
  // expected-warning {{ignoring 'shared_lock_function' attribute because its argument is invalid}}
int slf_function_bad_3() SHARED_LOCK_FUNCTION(muDoublePointer); // \
  // expected-warning {{'shared_lock_function' attribute requires arguments that are class type or point to class type}}
int slf_function_bad_4() SHARED_LOCK_FUNCTION(umu); // \
  // expected-warning {{'shared_lock_function' attribute requires arguments whose type is annotated with 'lockable' attribute}}

int slf_function_bad_1() SHARED_LOCK_FUNCTION(1); // \
  // expected-error {{'shared_lock_function' attribute parameter 1 is out of bounds: no parameters to index into}}
int slf_function_bad_5(Mutex x) SHARED_LOCK_FUNCTION(0); // \
  // expected-error {{'shared_lock_function' attribute parameter 1 is out of bounds: can only be 1, since there is one parameter}}
int slf_function_bad_6(Mutex x, Mutex y) SHARED_LOCK_FUNCTION(0); // \
  // expected-error {{'shared_lock_function' attribute parameter 1 is out of bounds: must be between 1 and 2}}
int slf_function_bad_7() SHARED_LOCK_FUNCTION(0); // \
  // expected-error {{'shared_lock_function' attribute parameter 1 is out of bounds: no parameters to index into}}


//-----------------------------------------//
//  Exclusive TryLock Function (etf)
//-----------------------------------------//

#if !__has_attribute(exclusive_trylock_function)
#error "Should support exclusive_trylock_function attribute"
#endif

// takes a mandatory boolean or integer argument specifying the retval
// plus an optional list of locks (vars/fields)

void etf_function() __attribute__((exclusive_trylock_function));  // \
  // expected-error {{attribute takes at least 1 argument}}

void etf_function_args() EXCLUSIVE_TRYLOCK_FUNCTION(1, mu2);

void etf_function_arg() EXCLUSIVE_TRYLOCK_FUNCTION(1);

int etf_testfn(int y) EXCLUSIVE_TRYLOCK_FUNCTION(1);

int etf_testfn(int y) {
  int x EXCLUSIVE_TRYLOCK_FUNCTION(1) = y; // \
    // expected-warning {{'exclusive_trylock_function' attribute only applies to functions and methods}}
  return x;
};

int etf_test_var EXCLUSIVE_TRYLOCK_FUNCTION(1); // \
  // expected-warning {{'exclusive_trylock_function' attribute only applies to functions and methods}}

class EtfFoo {
 private:
  int test_field EXCLUSIVE_TRYLOCK_FUNCTION(1); // \
    // expected-warning {{'exclusive_trylock_function' attribute only applies to functions and methods}}
  void test_method() EXCLUSIVE_TRYLOCK_FUNCTION(1);
};

class EXCLUSIVE_TRYLOCK_FUNCTION(1) EtfTestClass { // \
  // expected-warning {{'exclusive_trylock_function' attribute only applies to functions and methods}}
};

void etf_fun_params(int lvar EXCLUSIVE_TRYLOCK_FUNCTION(1)); // \
  // expected-warning {{'exclusive_trylock_function' attribute only applies to functions and methods}}

// Check argument parsing.

// legal attribute arguments
int etf_function_1() EXCLUSIVE_TRYLOCK_FUNCTION(1, muWrapper.mu);
int etf_function_2() EXCLUSIVE_TRYLOCK_FUNCTION(1, muDoubleWrapper.muWrapper->mu);
int etf_function_3() EXCLUSIVE_TRYLOCK_FUNCTION(1, muWrapper.getMu());
int etf_function_4() EXCLUSIVE_TRYLOCK_FUNCTION(1, *muWrapper.getMuPointer());
int etf_function_5() EXCLUSIVE_TRYLOCK_FUNCTION(1, &mu1);
int etf_function_6() EXCLUSIVE_TRYLOCK_FUNCTION(1, muRef);
int etf_function_7() EXCLUSIVE_TRYLOCK_FUNCTION(1, muDoubleWrapper.getWrapper()->getMu());
int etf_functetfn_8() EXCLUSIVE_TRYLOCK_FUNCTION(1, muPointer);
int etf_function_9() EXCLUSIVE_TRYLOCK_FUNCTION(true);


// illegal attribute arguments
int etf_function_bad_1() EXCLUSIVE_TRYLOCK_FUNCTION(mu1); // \
  // expected-error {{'exclusive_trylock_function' attribute first argument must be of int or bool type}}
int etf_function_bad_2() EXCLUSIVE_TRYLOCK_FUNCTION("mu"); // \
  // expected-error {{'exclusive_trylock_function' attribute first argument must be of int or bool type}}
int etf_function_bad_3() EXCLUSIVE_TRYLOCK_FUNCTION(muDoublePointer); // \
  // expected-error {{'exclusive_trylock_function' attribute first argument must be of int or bool type}}

int etf_function_bad_4() EXCLUSIVE_TRYLOCK_FUNCTION(1, "mu"); // \
  // expected-warning {{ignoring 'exclusive_trylock_function' attribute because its argument is invalid}}
int etf_function_bad_5() EXCLUSIVE_TRYLOCK_FUNCTION(1, muDoublePointer); // \
  // expected-warning {{'exclusive_trylock_function' attribute requires arguments that are class type or point to class type}}
int etf_function_bad_6() EXCLUSIVE_TRYLOCK_FUNCTION(1, umu); // \
  // expected-warning {{'exclusive_trylock_function' attribute requires arguments whose type is annotated with 'lockable' attribute}}


//-----------------------------------------//
//  Shared TryLock Function (stf)
//-----------------------------------------//

#if !__has_attribute(shared_trylock_function)
#error "Should support shared_trylock_function attribute"
#endif

// takes a mandatory boolean or integer argument specifying the retval
// plus an optional list of locks (vars/fields)

void stf_function() __attribute__((shared_trylock_function));  // \
  // expected-error {{attribute takes at least 1 argument}}

void stf_function_args() SHARED_TRYLOCK_FUNCTION(1, mu2);

void stf_function_arg() SHARED_TRYLOCK_FUNCTION(1);

int stf_testfn(int y) SHARED_TRYLOCK_FUNCTION(1);

int stf_testfn(int y) {
  int x SHARED_TRYLOCK_FUNCTION(1) = y; // \
    // expected-warning {{'shared_trylock_function' attribute only applies to functions and methods}}
  return x;
};

int stf_test_var SHARED_TRYLOCK_FUNCTION(1); // \
  // expected-warning {{'shared_trylock_function' attribute only applies to functions and methods}}

void stf_fun_params(int lvar SHARED_TRYLOCK_FUNCTION(1)); // \
  // expected-warning {{'shared_trylock_function' attribute only applies to functions and methods}}


class StfFoo {
 private:
  int test_field SHARED_TRYLOCK_FUNCTION(1); // \
    // expected-warning {{'shared_trylock_function' attribute only applies to functions and methods}}
  void test_method() SHARED_TRYLOCK_FUNCTION(1);
};

class SHARED_TRYLOCK_FUNCTION(1) StfTestClass { // \
    // expected-warning {{'shared_trylock_function' attribute only applies to functions and methods}}
};

// Check argument parsing.

// legal attribute arguments
int stf_function_1() SHARED_TRYLOCK_FUNCTION(1, muWrapper.mu);
int stf_function_2() SHARED_TRYLOCK_FUNCTION(1, muDoubleWrapper.muWrapper->mu);
int stf_function_3() SHARED_TRYLOCK_FUNCTION(1, muWrapper.getMu());
int stf_function_4() SHARED_TRYLOCK_FUNCTION(1, *muWrapper.getMuPointer());
int stf_function_5() SHARED_TRYLOCK_FUNCTION(1, &mu1);
int stf_function_6() SHARED_TRYLOCK_FUNCTION(1, muRef);
int stf_function_7() SHARED_TRYLOCK_FUNCTION(1, muDoubleWrapper.getWrapper()->getMu());
int stf_function_8() SHARED_TRYLOCK_FUNCTION(1, muPointer);
int stf_function_9() SHARED_TRYLOCK_FUNCTION(true);


// illegal attribute arguments
int stf_function_bad_1() SHARED_TRYLOCK_FUNCTION(mu1); // \
  // expected-error {{'shared_trylock_function' attribute first argument must be of int or bool type}}
int stf_function_bad_2() SHARED_TRYLOCK_FUNCTION("mu"); // \
  // expected-error {{'shared_trylock_function' attribute first argument must be of int or bool type}}
int stf_function_bad_3() SHARED_TRYLOCK_FUNCTION(muDoublePointer); // \
  // expected-error {{'shared_trylock_function' attribute first argument must be of int or bool type}}

int stf_function_bad_4() SHARED_TRYLOCK_FUNCTION(1, "mu"); // \
  // expected-warning {{ignoring 'shared_trylock_function' attribute because its argument is invalid}}
int stf_function_bad_5() SHARED_TRYLOCK_FUNCTION(1, muDoublePointer); // \
  // expected-warning {{'shared_trylock_function' attribute requires arguments that are class type or point to class type}}
int stf_function_bad_6() SHARED_TRYLOCK_FUNCTION(1, umu); // \
  // expected-warning {{'shared_trylock_function' attribute requires arguments whose type is annotated with 'lockable' attribute}}


//-----------------------------------------//
//  Unlock Function (uf)
//-----------------------------------------//

#if !__has_attribute(unlock_function)
#error "Should support unlock_function attribute"
#endif

// takes zero or more arguments, all locks (vars/fields)

void uf_function() UNLOCK_FUNCTION();

void uf_function_args() UNLOCK_FUNCTION(mu1, mu2);

int uf_testfn(int y) UNLOCK_FUNCTION();

int uf_testfn(int y) {
  int x UNLOCK_FUNCTION() = y; // \
    // expected-warning {{'unlock_function' attribute only applies to functions and methods}}
  return x;
};

int uf_test_var UNLOCK_FUNCTION(); // \
  // expected-warning {{'unlock_function' attribute only applies to functions and methods}}

class UfFoo {
 private:
  int test_field UNLOCK_FUNCTION(); // \
    // expected-warning {{'unlock_function' attribute only applies to functions and methods}}
  void test_method() UNLOCK_FUNCTION();
};

class NO_THREAD_SAFETY_ANALYSIS UfTestClass { // \
  // expected-warning {{'no_thread_safety_analysis' attribute only applies to functions and methods}}
};

void uf_fun_params(int lvar UNLOCK_FUNCTION()); // \
  // expected-warning {{'unlock_function' attribute only applies to functions and methods}}

// Check argument parsing.

// legal attribute arguments
int uf_function_1() UNLOCK_FUNCTION(muWrapper.mu);
int uf_function_2() UNLOCK_FUNCTION(muDoubleWrapper.muWrapper->mu);
int uf_function_3() UNLOCK_FUNCTION(muWrapper.getMu());
int uf_function_4() UNLOCK_FUNCTION(*muWrapper.getMuPointer());
int uf_function_5() UNLOCK_FUNCTION(&mu1);
int uf_function_6() UNLOCK_FUNCTION(muRef);
int uf_function_7() UNLOCK_FUNCTION(muDoubleWrapper.getWrapper()->getMu());
int uf_function_8() UNLOCK_FUNCTION(muPointer);
int uf_function_9(Mutex x) UNLOCK_FUNCTION(1);
int uf_function_9(Mutex x, Mutex y) UNLOCK_FUNCTION(1,2);


// illegal attribute arguments
int uf_function_bad_2() UNLOCK_FUNCTION("mu"); // \
  // expected-warning {{ignoring 'unlock_function' attribute because its argument is invalid}}
int uf_function_bad_3() UNLOCK_FUNCTION(muDoublePointer); // \
  // expected-warning {{'unlock_function' attribute requires arguments that are class type or point to class type}}
int uf_function_bad_4() UNLOCK_FUNCTION(umu); // \
  // expected-warning {{'unlock_function' attribute requires arguments whose type is annotated with 'lockable' attribute}}

int uf_function_bad_1() UNLOCK_FUNCTION(1); // \
  // expected-error {{'unlock_function' attribute parameter 1 is out of bounds: no parameters to index into}}
int uf_function_bad_5(Mutex x) UNLOCK_FUNCTION(0); // \
  // expected-error {{'unlock_function' attribute parameter 1 is out of bounds: can only be 1, since there is one parameter}}
int uf_function_bad_6(Mutex x, Mutex y) UNLOCK_FUNCTION(0); // \
  // expected-error {{'unlock_function' attribute parameter 1 is out of bounds: must be between 1 and 2}}
int uf_function_bad_7() UNLOCK_FUNCTION(0); // \
  // expected-error {{'unlock_function' attribute parameter 1 is out of bounds: no parameters to index into}}


//-----------------------------------------//
//  Lock Returned (lr)
//-----------------------------------------//

#if !__has_attribute(lock_returned)
#error "Should support lock_returned attribute"
#endif

// Takes exactly one argument, a var/field

void lr_function() __attribute__((lock_returned)); // \
  // expected-error {{attribute takes one argument}}

void lr_function_arg() LOCK_RETURNED(mu1);

void lr_function_args() __attribute__((lock_returned(mu1, mu2))); // \
  // expected-error {{attribute takes one argument}}

int lr_testfn(int y) LOCK_RETURNED(mu1);

int lr_testfn(int y) {
  int x LOCK_RETURNED(mu1) = y; // \
    // expected-warning {{'lock_returned' attribute only applies to functions and methods}}
  return x;
};

int lr_test_var LOCK_RETURNED(mu1); // \
  // expected-warning {{'lock_returned' attribute only applies to functions and methods}}

void lr_fun_params(int lvar LOCK_RETURNED(mu1)); // \
  // expected-warning {{'lock_returned' attribute only applies to functions and methods}}

class LrFoo {
 private:
  int test_field LOCK_RETURNED(mu1); // \
    // expected-warning {{'lock_returned' attribute only applies to functions and methods}}
  void test_method() LOCK_RETURNED(mu1);
};

class LOCK_RETURNED(mu1) LrTestClass { // \
    // expected-warning {{'lock_returned' attribute only applies to functions and methods}}
};

// Check argument parsing.

// legal attribute arguments
int lr_function_1() LOCK_RETURNED(muWrapper.mu);
int lr_function_2() LOCK_RETURNED(muDoubleWrapper.muWrapper->mu);
int lr_function_3() LOCK_RETURNED(muWrapper.getMu());
int lr_function_4() LOCK_RETURNED(*muWrapper.getMuPointer());
int lr_function_5() LOCK_RETURNED(&mu1);
int lr_function_6() LOCK_RETURNED(muRef);
int lr_function_7() LOCK_RETURNED(muDoubleWrapper.getWrapper()->getMu());
int lr_function_8() LOCK_RETURNED(muPointer);


// illegal attribute arguments
int lr_function_bad_1() LOCK_RETURNED(1); // \
  // expected-warning {{'lock_returned' attribute requires arguments that are class type or point to class type}}
int lr_function_bad_2() LOCK_RETURNED("mu"); // \
  // expected-warning {{ignoring 'lock_returned' attribute because its argument is invalid}}
int lr_function_bad_3() LOCK_RETURNED(muDoublePointer); // \
  // expected-warning {{'lock_returned' attribute requires arguments that are class type or point to class type}}
int lr_function_bad_4() LOCK_RETURNED(umu); // \
  // expected-warning {{'lock_returned' attribute requires arguments whose type is annotated with 'lockable' attribute}}



//-----------------------------------------//
//  Locks Excluded (le)
//-----------------------------------------//

#if !__has_attribute(locks_excluded)
#error "Should support locks_excluded attribute"
#endif

// takes one or more arguments, all locks (vars/fields)

void le_function() __attribute__((locks_excluded)); // \
  // expected-error {{attribute takes at least 1 argument}}

void le_function_arg() LOCKS_EXCLUDED(mu1);

void le_function_args() LOCKS_EXCLUDED(mu1, mu2);

int le_testfn(int y) LOCKS_EXCLUDED(mu1);

int le_testfn(int y) {
  int x LOCKS_EXCLUDED(mu1) = y; // \
    // expected-warning {{'locks_excluded' attribute only applies to functions and methods}}
  return x;
};

int le_test_var LOCKS_EXCLUDED(mu1); // \
  // expected-warning {{'locks_excluded' attribute only applies to functions and methods}}

void le_fun_params(int lvar LOCKS_EXCLUDED(mu1)); // \
  // expected-warning {{'locks_excluded' attribute only applies to functions and methods}}

class LeFoo {
 private:
  int test_field LOCKS_EXCLUDED(mu1); // \
    // expected-warning {{'locks_excluded' attribute only applies to functions and methods}}
  void test_method() LOCKS_EXCLUDED(mu1);
};

class LOCKS_EXCLUDED(mu1) LeTestClass { // \
  // expected-warning {{'locks_excluded' attribute only applies to functions and methods}}
};

// Check argument parsing.

// legal attribute arguments
int le_function_1() LOCKS_EXCLUDED(muWrapper.mu);
int le_function_2() LOCKS_EXCLUDED(muDoubleWrapper.muWrapper->mu);
int le_function_3() LOCKS_EXCLUDED(muWrapper.getMu());
int le_function_4() LOCKS_EXCLUDED(*muWrapper.getMuPointer());
int le_function_5() LOCKS_EXCLUDED(&mu1);
int le_function_6() LOCKS_EXCLUDED(muRef);
int le_function_7() LOCKS_EXCLUDED(muDoubleWrapper.getWrapper()->getMu());
int le_function_8() LOCKS_EXCLUDED(muPointer);


// illegal attribute arguments
int le_function_bad_1() LOCKS_EXCLUDED(1); // \
  // expected-warning {{'locks_excluded' attribute requires arguments that are class type or point to class type}}
int le_function_bad_2() LOCKS_EXCLUDED("mu"); // \
  // expected-warning {{ignoring 'locks_excluded' attribute because its argument is invalid}}
int le_function_bad_3() LOCKS_EXCLUDED(muDoublePointer); // \
  // expected-warning {{'locks_excluded' attribute requires arguments that are class type or point to class type}}
int le_function_bad_4() LOCKS_EXCLUDED(umu); // \
  // expected-warning {{'locks_excluded' attribute requires arguments whose type is annotated with 'lockable' attribute}}



//-----------------------------------------//
//  Exclusive Locks Required (elr)
//-----------------------------------------//

#if !__has_attribute(exclusive_locks_required)
#error "Should support exclusive_locks_required attribute"
#endif

// takes one or more arguments, all locks (vars/fields)

void elr_function() __attribute__((exclusive_locks_required)); // \
  // expected-error {{attribute takes at least 1 argument}}

void elr_function_arg() EXCLUSIVE_LOCKS_REQUIRED(mu1);

void elr_function_args() EXCLUSIVE_LOCKS_REQUIRED(mu1, mu2);

int elr_testfn(int y) EXCLUSIVE_LOCKS_REQUIRED(mu1);

int elr_testfn(int y) {
  int x EXCLUSIVE_LOCKS_REQUIRED(mu1) = y; // \
    // expected-warning {{'exclusive_locks_required' attribute only applies to functions and methods}}
  return x;
};

int elr_test_var EXCLUSIVE_LOCKS_REQUIRED(mu1); // \
  // expected-warning {{'exclusive_locks_required' attribute only applies to functions and methods}}

void elr_fun_params(int lvar EXCLUSIVE_LOCKS_REQUIRED(mu1)); // \
  // expected-warning {{'exclusive_locks_required' attribute only applies to functions and methods}}

class ElrFoo {
 private:
  int test_field EXCLUSIVE_LOCKS_REQUIRED(mu1); // \
    // expected-warning {{'exclusive_locks_required' attribute only applies to functions and methods}}
  void test_method() EXCLUSIVE_LOCKS_REQUIRED(mu1);
};

class EXCLUSIVE_LOCKS_REQUIRED(mu1) ElrTestClass { // \
  // expected-warning {{'exclusive_locks_required' attribute only applies to functions and methods}}
};

// Check argument parsing.

// legal attribute arguments
int elr_function_1() EXCLUSIVE_LOCKS_REQUIRED(muWrapper.mu);
int elr_function_2() EXCLUSIVE_LOCKS_REQUIRED(muDoubleWrapper.muWrapper->mu);
int elr_function_3() EXCLUSIVE_LOCKS_REQUIRED(muWrapper.getMu());
int elr_function_4() EXCLUSIVE_LOCKS_REQUIRED(*muWrapper.getMuPointer());
int elr_function_5() EXCLUSIVE_LOCKS_REQUIRED(&mu1);
int elr_function_6() EXCLUSIVE_LOCKS_REQUIRED(muRef);
int elr_function_7() EXCLUSIVE_LOCKS_REQUIRED(muDoubleWrapper.getWrapper()->getMu());
int elr_function_8() EXCLUSIVE_LOCKS_REQUIRED(muPointer);


// illegal attribute arguments
int elr_function_bad_1() EXCLUSIVE_LOCKS_REQUIRED(1); // \
  // expected-warning {{'exclusive_locks_required' attribute requires arguments that are class type or point to class type}}
int elr_function_bad_2() EXCLUSIVE_LOCKS_REQUIRED("mu"); // \
  // expected-warning {{ignoring 'exclusive_locks_required' attribute because its argument is invalid}}
int elr_function_bad_3() EXCLUSIVE_LOCKS_REQUIRED(muDoublePointer); // \
  // expected-warning {{'exclusive_locks_required' attribute requires arguments that are class type or point to class type}}
int elr_function_bad_4() EXCLUSIVE_LOCKS_REQUIRED(umu); // \
  // expected-warning {{'exclusive_locks_required' attribute requires arguments whose type is annotated with 'lockable' attribute}}




//-----------------------------------------//
//  Shared Locks Required (slr)
//-----------------------------------------//

#if !__has_attribute(shared_locks_required)
#error "Should support shared_locks_required attribute"
#endif

// takes one or more arguments, all locks (vars/fields)

void slr_function() __attribute__((shared_locks_required)); // \
  // expected-error {{attribute takes at least 1 argument}}

void slr_function_arg() SHARED_LOCKS_REQUIRED(mu1);

void slr_function_args() SHARED_LOCKS_REQUIRED(mu1, mu2);

int slr_testfn(int y) SHARED_LOCKS_REQUIRED(mu1);

int slr_testfn(int y) {
  int x SHARED_LOCKS_REQUIRED(mu1) = y; // \
    // expected-warning {{'shared_locks_required' attribute only applies to functions and methods}}
  return x;
};

int slr_test_var SHARED_LOCKS_REQUIRED(mu1); // \
  // expected-warning {{'shared_locks_required' attribute only applies to functions and methods}}

void slr_fun_params(int lvar SHARED_LOCKS_REQUIRED(mu1)); // \
  // expected-warning {{'shared_locks_required' attribute only applies to functions and methods}}

class SlrFoo {
 private:
  int test_field SHARED_LOCKS_REQUIRED(mu1); // \
    // expected-warning {{'shared_locks_required' attribute only applies to functions and methods}}
  void test_method() SHARED_LOCKS_REQUIRED(mu1);
};

class SHARED_LOCKS_REQUIRED(mu1) SlrTestClass { // \
  // expected-warning {{'shared_locks_required' attribute only applies to functions and methods}}
};

// Check argument parsing.

// legal attribute arguments
int slr_function_1() SHARED_LOCKS_REQUIRED(muWrapper.mu);
int slr_function_2() SHARED_LOCKS_REQUIRED(muDoubleWrapper.muWrapper->mu);
int slr_function_3() SHARED_LOCKS_REQUIRED(muWrapper.getMu());
int slr_function_4() SHARED_LOCKS_REQUIRED(*muWrapper.getMuPointer());
int slr_function_5() SHARED_LOCKS_REQUIRED(&mu1);
int slr_function_6() SHARED_LOCKS_REQUIRED(muRef);
int slr_function_7() SHARED_LOCKS_REQUIRED(muDoubleWrapper.getWrapper()->getMu());
int slr_function_8() SHARED_LOCKS_REQUIRED(muPointer);


// illegal attribute arguments
int slr_function_bad_1() SHARED_LOCKS_REQUIRED(1); // \
  // expected-warning {{'shared_locks_required' attribute requires arguments that are class type or point to class type}}
int slr_function_bad_2() SHARED_LOCKS_REQUIRED("mu"); // \
  // expected-warning {{ignoring 'shared_locks_required' attribute because its argument is invalid}}
int slr_function_bad_3() SHARED_LOCKS_REQUIRED(muDoublePointer); // \
  // expected-warning {{'shared_locks_required' attribute requires arguments that are class type or point to class type}}
int slr_function_bad_4() SHARED_LOCKS_REQUIRED(umu); // \
  // expected-warning {{'shared_locks_required' attribute requires arguments whose type is annotated with 'lockable' attribute}}


//-----------------------------------------//
//  Regression tests for unusual cases.
//-----------------------------------------//

int trivially_false_edges(bool b) {
  // Create NULL (never taken) edges in CFG
  if (false) return 1;
  else       return 2;
}

// Possible Clang bug -- method pointer in template parameter
class UnFoo {
public:
  void foo();
};

template<void (UnFoo::*methptr)()>
class MCaller {
public:
  static void call_method_ptr(UnFoo *f) {
    // FIXME: Possible Clang bug:
    // getCalleeDecl() returns NULL in the following case:
    (f->*methptr)();
  }
};

void call_method_ptr_inst(UnFoo* f) {
  MCaller<&UnFoo::foo>::call_method_ptr(f);
}

int temp;
void empty_back_edge() {
  // Create a back edge to a block with with no statements
  for (;;) {
    ++temp;
    if (temp > 10) break;
  }
}

struct Foomger {
  void operator++();
};

struct Foomgoper {
  Foomger f;

  bool done();
  void invalid_back_edge() {
    do {
      // FIXME: Possible Clang bug:
      // The first statement in this basic block has no source location
      ++f;
    } while (!done());
  }
};


//-----------------------------------------------------
// Parsing of member variables and function parameters
//------------------------------------------------------

Mutex gmu;

class StaticMu {
  static Mutex statmu;
};

class FooLate {
public:
  void foo1()           EXCLUSIVE_LOCKS_REQUIRED(gmu)   { }
  void foo2()           EXCLUSIVE_LOCKS_REQUIRED(mu)    { }
  void foo3(Mutex *m)   EXCLUSIVE_LOCKS_REQUIRED(m)     { }
  void foo3(FooLate *f) EXCLUSIVE_LOCKS_REQUIRED(f->mu) { }
  void foo4(FooLate *f) EXCLUSIVE_LOCKS_REQUIRED(f->mu);

  static void foo5()    EXCLUSIVE_LOCKS_REQUIRED(mu); // \
    // expected-error {{invalid use of member 'mu' in static member function}}

  template <class T>
  void foo6() EXCLUSIVE_LOCKS_REQUIRED(T::statmu) { }

  template <class T>
  void foo7(T* f) EXCLUSIVE_LOCKS_REQUIRED(f->mu) { }

  int a GUARDED_BY(gmu);
  int b GUARDED_BY(mu);
  int c GUARDED_BY(this->mu);

  Mutex mu;
};

//-------------------------
// Empty argument lists
//-------------------------

class LOCKABLE EmptyArgListsTest {
  void lock() EXCLUSIVE_LOCK_FUNCTION() { }
  void unlock() UNLOCK_FUNCTION() { }
};


namespace FunctionDefinitionParseTest {
// Test parsing of attributes on function definitions.

class Foo {
public:
  Mutex mu_;
  void foo1();
  void foo2(Foo *f);
};

template <class T>
class Bar {
public:
  Mutex mu_;
  void bar();
};

void Foo::foo1()       EXCLUSIVE_LOCKS_REQUIRED(mu_) { }
void Foo::foo2(Foo *f) EXCLUSIVE_LOCKS_REQUIRED(f->mu_) { }

template <class T>
void Bar<T>::bar() EXCLUSIVE_LOCKS_REQUIRED(mu_) { }

void baz(Foo *f) EXCLUSIVE_LOCKS_REQUIRED(f->mu_) { }

} // end namespace


namespace TestMultiDecl {

class Foo {
public:
  int GUARDED_BY(mu_) a;
  int GUARDED_BY(mu_) b, c;

private:
  Mutex mu_;
};

} // end namespace TestMultiDecl


namespace NestedClassLateDecl {

class Foo {
  class Bar {
    int a GUARDED_BY(mu);
    int b GUARDED_BY(fooMuStatic);

    void bar()        EXCLUSIVE_LOCKS_REQUIRED(mu)       { a = 0;    }
    void bar2(Bar* b) EXCLUSIVE_LOCKS_REQUIRED(b->mu)    { b->a = 0; }
    void bar3(Foo* f) EXCLUSIVE_LOCKS_REQUIRED(f->fooMu) { f->a = 0; }

    Mutex mu;
  };

  int a GUARDED_BY(fooMu);
  Mutex fooMu;
  static Mutex fooMuStatic;
};

}

namespace PointerToMemberTest {

// Empty string should be ignored.
int  testEmptyAttribute GUARDED_BY("");
void testEmptyAttributeFunction() EXCLUSIVE_LOCKS_REQUIRED("");

class Graph {
public:
  Mutex mu_;

  static Mutex* get_static_mu() LOCK_RETURNED(&Graph::mu_);
};

class Node {
public:
  void foo() EXCLUSIVE_LOCKS_REQUIRED(&Graph::mu_);
  int a GUARDED_BY(&Graph::mu_);
};

}


namespace SmartPointerTest {

template<class T>
class smart_ptr {
 public:
  T* operator->() { return ptr_; }
  T& operator*()  { return ptr_; }

 private:
  T* ptr_;
};


Mutex gmu;
smart_ptr<int> gdat PT_GUARDED_BY(gmu);


class MyClass {
public:
  Mutex mu_;
  smart_ptr<Mutex> smu_;


  smart_ptr<int> a PT_GUARDED_BY(mu_);
  int b            GUARDED_BY(smu_);
};

}


namespace InheritanceTest {

class LOCKABLE Base {
 public:
  void lock()   EXCLUSIVE_LOCK_FUNCTION();
  void unlock() UNLOCK_FUNCTION();
};

class Base2 { };

class Derived1 : public Base { };

class Derived2 : public Base2, public Derived1 { };

class Derived3 : public Base2 { };

class Foo {
  Derived1 mu1_;
  Derived2 mu2_;
  Derived3 mu3_;
  int a GUARDED_BY(mu1_);
  int b GUARDED_BY(mu2_);
  int c GUARDED_BY(mu3_);  // \
    // expected-warning {{'guarded_by' attribute requires arguments whose type is annotated with 'lockable' attribute; type here is 'class InheritanceTest::Derived3'}}

  void foo() EXCLUSIVE_LOCKS_REQUIRED(mu1_, mu2_) {
    a = 0;
    b = 0;
  }
};

}


namespace InvalidDeclTest {

class Foo { };
namespace {
void Foo::bar(Mutex* mu) LOCKS_EXCLUDED(mu) { } // \
   // expected-error   {{cannot define or redeclare 'bar' here because namespace '' does not enclose namespace 'Foo'}} \
   // expected-warning {{attribute locks_excluded ignored, because it is not attached to a declaration}}
}

} // end namespace InvalidDeclTest


namespace StaticScopeTest {

class FooStream;

class Foo {
  mutable Mutex mu;
  int a GUARDED_BY(mu);

  static int si GUARDED_BY(mu); // \
    // expected-error {{invalid use of non-static data member 'mu'}}

  static void foo() EXCLUSIVE_LOCKS_REQUIRED(mu); // \
    // expected-error {{invalid use of member 'mu' in static member function}}

  friend FooStream& operator<<(FooStream& s, const Foo& f)
    EXCLUSIVE_LOCKS_REQUIRED(mu); // \
    // expected-error {{invalid use of non-static data member 'mu'}}
};


} // end namespace StaticScopeTest


namespace FunctionAttributesInsideClass_ICE_Test {

class Foo {
public:
  /*  Originally found when parsing foo() as an ordinary method after the
   *  the following:

  template <class T>
  void syntaxErrorMethod(int i) {
    if (i) {
      foo(
    }
  }
  */

  void method() {
    void foo() EXCLUSIVE_LOCKS_REQUIRED(mu); // \
      // expected-error {{use of undeclared identifier 'mu'}}
  }
};

}  // end namespace FunctionAttributesInsideClass_ICE_Test

