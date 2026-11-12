// RUN: %clang_analyze_cc1 -analyzer-checker=core,osx.cocoa.RetainCount,debug.ExprInspection -verify -Wno-objc-root-class %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,osx.cocoa.RetainCount,debug.ExprInspection -verify -Wno-objc-root-class -fobjc-arc %s

void clang_analyzer_eval(bool);
void clang_analyzer_checkInlined(bool);

@interface IntWrapper
@property (readonly) int &value;
@end

@implementation IntWrapper
@synthesize value;
@end

void testReferenceConsistency(IntWrapper *w) {
  clang_analyzer_eval(w.value == w.value); // expected-warning{{TRUE}}
  clang_analyzer_eval(&w.value == &w.value); // expected-warning{{TRUE}}

  if (w.value != 42)
    return;

  clang_analyzer_eval(w.value == 42); // expected-warning{{TRUE}}
}

void testReferenceAssignment(IntWrapper *w) {
  w.value = 42;
  clang_analyzer_eval(w.value == 42); // expected-warning{{TRUE}}
}


struct IntWrapperStruct {
  int value;
};

@interface StructWrapper
@property IntWrapperStruct inner;
@end

@implementation StructWrapper
@synthesize inner;
@end

void testConsistencyStruct(StructWrapper *w) {
  clang_analyzer_eval(w.inner.value == w.inner.value); // expected-warning{{TRUE}}

  int origValue = w.inner.value;
  if (origValue != 42)
    return;

  clang_analyzer_eval(w.inner.value == 42); // expected-warning{{TRUE}}
}


class CustomCopy {
public:
  CustomCopy() : value(0) {}
  CustomCopy(const CustomCopy &other) : value(other.value) {
    clang_analyzer_checkInlined(true); // expected-warning{{TRUE}}
  }
  int value;
};

@interface CustomCopyWrapper
@property CustomCopy inner;
@end

@implementation CustomCopyWrapper
//@synthesize inner;
@end

void testConsistencyCustomCopy(CustomCopyWrapper *w) {
  clang_analyzer_eval(w.inner.value == w.inner.value); // expected-warning{{TRUE}}

  int origValue = w.inner.value;
  if (origValue != 42)
    return;

  clang_analyzer_eval(w.inner.value == 42); // expected-warning{{TRUE}}
}

@protocol NoDirectPropertyDecl
@property IntWrapperStruct inner;
@end
@interface NoDirectPropertyDecl <NoDirectPropertyDecl>
@end
@implementation NoDirectPropertyDecl
@synthesize inner;
@end

void testNoDirectPropertyDecl(NoDirectPropertyDecl *w) {
  clang_analyzer_eval(w.inner.value == w.inner.value); // expected-warning{{TRUE}}

  int origValue = w.inner.value;
  if (origValue != 42)
    return;

  clang_analyzer_eval(w.inner.value == 42); // expected-warning{{TRUE}}
}
