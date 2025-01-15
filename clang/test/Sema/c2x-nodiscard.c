// RUN: %clang_cc1 -fsyntax-only -std=c2x -verify %s

// This is the latest version of nodiscard that we support.
_Static_assert(__has_c_attribute(nodiscard) == 202003L);

struct [[nodiscard]] S1 { // ok
  int i;
};
struct [[nodiscard, nodiscard]] S2 { // ok
  int i;
};
struct [[nodiscard("Wrong")]] S3 {
  int i;
};

struct S3 get_s3(void);

[[nodiscard]] int f1(void);
enum [[nodiscard]] E1 { One };

[[nodiscard]] int i; // expected-warning {{'nodiscard' attribute only applies to Objective-C methods, enums, structs, unions, classes, functions, function pointers, and typedefs}}

struct [[nodiscard]] S4 {
  int i;
};
struct S4 get_s(void);

enum [[nodiscard]] E2 { Two };
enum E2 get_e(void);

[[nodiscard]] int get_i(void);

void f2(void) {
  get_s(); // expected-warning {{ignoring return value of type 'S4' declared with 'nodiscard' attribute}}
  get_s3(); // expected-warning {{ignoring return value of type 'S3' declared with 'nodiscard' attribute: Wrong}}
  get_i(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  get_e(); // expected-warning {{ignoring return value of type 'E2' declared with 'nodiscard' attribute}}

  // Okay, warnings are not encouraged
  (void)get_s();
  (void)get_s3();
  (void)get_i();
  (void)get_e();
}

struct [[nodiscard]] error_info{
  int i;
};

struct error_info enable_missile_safety_mode(void);
void launch_missiles(void);
void test_missiles(void) {
  enable_missile_safety_mode(); // expected-warning {{ignoring return value of type 'error_info' declared with 'nodiscard'}}
  launch_missiles();
}

[[nodiscard]] int f3();

void GH104391() {
#define M (unsigned int) f3()
  M; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
