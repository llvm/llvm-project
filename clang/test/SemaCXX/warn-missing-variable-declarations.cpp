// RUN: %clang -Wmissing-variable-declarations -fsyntax-only -Xclang -verify %s

// Variable declarations that should trigger a warning.
int vbad1; // expected-warning{{no previous extern declaration for non-static variable 'vbad1'}}
int vbad2 = 10; // expected-warning{{no previous extern declaration for non-static variable 'vbad2'}}

namespace x {
  int vbad3; // expected-warning{{no previous extern declaration for non-static variable 'vbad3'}}
}

// Variable declarations that should not trigger a warning.
static int vgood1;
extern int vgood2;
int vgood2;
static struct {
  int mgood1;
} vgood3;

// Functions should never trigger a warning.
void fgood1(void);
void fgood2(void) {
  int lgood1;
  static int lgood2;
}
static void fgood3(void) {
  int lgood3;
  static int lgood4;
}

// Structures, namespaces and classes should be unaffected.
struct sgood1 {
  int mgood2;
};
struct {
  int mgood3;
} sgood2;
class CGood1 {
  static int MGood1;
};
int CGood1::MGood1;
namespace {
  int mgood4;
}
