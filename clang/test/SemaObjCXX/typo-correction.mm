// RUN: %clang_cc1 %s -verify -fsyntax-only -Wno-objc-root-class

class ClassA {};

class ClassB {
public:
  ClassB(ClassA* parent=0);
  ~ClassB();
};

@interface NSObject
@end

@interface InterfaceA : NSObject
@property(nonatomic, assign) ClassA *m_prop1; // expected-note {{here}}
@property(nonatomic, assign) ClassB *m_prop2;
@end

@implementation InterfaceA
- (id)test {
  self.m_prop2 = new ClassB(m_prop1); // expected-error {{use of undeclared identifier 'm_prop1'; did you mean '_m_prop1'?}}
}
@end

@interface InvalidNameInIvarAndPropertyBase
{
@public
  float  _a;
}
@property float _b;
@end

void invalidNameInIvarAndPropertyBase() {
  float a = ((InvalidNameInIvarAndPropertyBase*)node)->_a; // expected-error {{use of undeclared identifier 'node'}}
  float b = ((InvalidNameInIvarAndPropertyBase*)node)._b; // expected-error {{use of undeclared identifier 'node'}}
}

// Typo correction for a property when it has as correction candidates
// synthesized ivar and a class name, both at the same edit distance.
@class TypoCandidate;

@interface PropertyType : NSObject
@property int x;
@end

@interface InterfaceC : NSObject
@property(assign) PropertyType *typoCandidate; // expected-note {{'_typoCandidate' declared here}}
@end

@implementation InterfaceC
-(void)method {
  typoCandidate.x = 0; // expected-error {{use of undeclared identifier 'typoCandidate'; did you mean '_typoCandidate'?}}
}
@end

// The scope of 'do-while' ends before typo-correction takes place.

struct Mat2 { int rows; };

@implementation ImplNoInt // expected-warning {{cannot find interface declaration for 'ImplNoInt'}}

- (void)typoCorrentInDoWhile {
  Mat2 tlMat1; // expected-note {{'tlMat1' declared here}}
  // Create many scopes to exhaust the cache.
  do {
    for (int index = 0; index < 2; index++) {
      if (true) {
        for (int specialTileType = 1; specialTileType < 5; specialTileType++) {
          for (int i = 0; i < 10; i++) {
            for (double scale = 0.95; scale <= 1.055; scale += 0.05) {
              for (int j = 0; j < 10; j++) {
                if (1 > 0.9) {
                    for (int sptile = 1; sptile < 5; sptile++) {
                    }
                }
              }
            }
          }
        }
      }
    }
  } while (tlMat.rows); // expected-error {{use of undeclared identifier 'tlMat'; did you mean 'tlMat1'}}
}

@end
