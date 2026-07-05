// Note: the run lines follow their respective tests, since line/column
// matter in this test.

@interface Foo  {
  void *isa;
}
@property(copy) Foo *myprop;
@property(retain, nonatomic) id xx;

// RUN: c-index-test -code-completion-at=%s:7:11 %s -fobjc-runtime=macosx-10.4 -fno-objc-arc | FileCheck -check-prefix=CHECK-CC1 -check-prefix=CHECK-CC1-NOWEAK %s
// RUN: c-index-test -code-completion-at=%s:7:11 %s -fobjc-runtime=macosx-10.8 -Xclang -fobjc-weak -fno-objc-arc | FileCheck -check-prefix=CHECK-CC1 -check-prefix=CHECK-CC1-WEAK  %s
// CHECK-CC1: {TypedText assign}
// CHECK-CC1-NEXT: {TypedText atomic}
// CHECK-CC1-NEXT: {TypedText copy}
// CHECK-CC1-NEXT: {TypedText getter}{Text =}{Placeholder method}
// CHECK-CC1-NEXT: {TypedText nonatomic}
// CHECK-CC1: {TypedText nonnull}
// CHECK-CC1-NEXT: {TypedText null_resettable}
// CHECK-CC1-NEXT: {TypedText null_unspecified}
// CHECK-CC1-NEXT: {TypedText nullable}
// CHECK-CC1-NEXT: {TypedText readonly}
// CHECK-CC1-NEXT: {TypedText readwrite}
// CHECK-CC1-NEXT: {TypedText retain}
// CHECK-CC1-NEXT: {TypedText setter}{Text =}{Placeholder method}
// CHECK-CC1-NEXT: {TypedText strong}
// CHECK-CC1-NEXT: {TypedText unsafe_unretained}
// CHECK-CC1-NOWEAK-NOT: {TypedText weak}
// CHECK-CC1-WEAK-NEXT: {TypedText weak}

// RUN: c-index-test -code-completion-at=%s:7:11 %s -fobjc-arc -fobjc-runtime=macosx-10.7 | FileCheck -check-prefix=CHECK-CC1-ARC %s
// CHECK-CC1-ARC: {TypedText assign}
// CHECK-CC1-ARC-NEXT: {TypedText atomic}
// CHECK-CC1-ARC-NEXT: {TypedText copy}
// CHECK-CC1-ARC-NEXT: {TypedText getter}{Text =}{Placeholder method}
// CHECK-CC1-ARC-NEXT: {TypedText nonatomic}
// CHECK-CC1-ARC-NEXT: {TypedText nonnull}
// CHECK-CC1-ARC-NEXT: {TypedText null_resettable}
// CHECK-CC1-ARC-NEXT: {TypedText null_unspecified}
// CHECK-CC1-ARC-NEXT: {TypedText nullable}
// CHECK-CC1-ARC-NEXT: {TypedText readonly}
// CHECK-CC1-ARC-NEXT: {TypedText readwrite}
// CHECK-CC1-ARC-NEXT: {TypedText retain}
// CHECK-CC1-ARC-NEXT: {TypedText setter}{Text =}{Placeholder method}
// CHECK-CC1-ARC-NEXT: {TypedText strong}
// CHECK-CC1-ARC-NEXT: {TypedText unsafe_unretained}
// CHECK-CC1-ARC-NEXT: {TypedText weak}

// RUN: c-index-test -code-completion-at=%s:8:18 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: {TypedText getter}{Text =}{Placeholder method}
// CHECK-CC2-NEXT: {TypedText nonatomic}
// CHECK-CC2-NEXT: {TypedText nonnull}
// CHECK-CC2-NEXT: {TypedText null_resettable}
// CHECK-CC2-NEXT: {TypedText null_unspecified}
// CHECK-CC2-NEXT: {TypedText nullable}
// CHECK-CC2-NEXT: {TypedText readonly}
// CHECK-CC2-NEXT: {TypedText readwrite}
// CHECK-CC2-NEXT: {TypedText setter}{Text =}{Placeholder method}
@end
