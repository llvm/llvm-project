// RUN: not %clang_cc1 -no-opaque-pointers -emit-llvm -fobjc-arc -triple x86_64-apple-darwin10 %s -DPROTOEXPR -o - 2>&1 \
// RUN:     | FileCheck -check-prefix=PROTOEXPR %s

// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -fobjc-arc -triple x86_64-apple-darwin10 %s -DREDUNDANCY -o - \
// RUN:     | FileCheck -check-prefix=REDUNDANCY1 %s
// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -fobjc-arc -triple x86_64-apple-darwin10 %s -DREDUNDANCY -o - \
// RUN:     | FileCheck -check-prefix=REDUNDANCY2 %s

// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -fobjc-arc -triple x86_64-apple-darwin10 %s -DBASE -o - \
// RUN:     | FileCheck -check-prefix=NONFRAGILE %s
// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -fobjc-arc -triple x86_64-apple-darwin10 %s -DINHERITANCE -o - \
// RUN:     | FileCheck -check-prefix=INHERITANCE %s

// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -triple x86_64-apple-darwin -fobjc-runtime=macosx-fragile-10.5 %s -DBASE -o - \
// RUN:     | FileCheck -check-prefix=FRAGILE %s
// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -triple x86_64-apple-darwin -fobjc-runtime=macosx-fragile-10.5 %s -DINHERITANCE -o - \
// RUN:     | FileCheck -check-prefix=FRAGILEINHERITANCE %s

// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -triple x86_64-linux-gnu -fobjc-runtime=gnustep %s -DBASE -o - \
// RUN:     | FileCheck -check-prefix=GNU %s
// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -triple x86_64-linux-gnu -fobjc-runtime=gnustep %s -DINHERITANCE -o - \
// RUN:     | FileCheck -check-prefix=GNUINHERITANCE %s
//
// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -triple x86_64-linux-gnu -fobjc-runtime=gnustep-2 %s -DBASE -o - \
// RUN:     | FileCheck -check-prefix=GNU2 %s
// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm -triple x86_64-linux-gnu -fobjc-runtime=gnustep-2 %s -DINHERITANCE -o - \
// RUN:     | FileCheck -check-prefix=GNU2INHERITANCE %s

__attribute__((objc_root_class))
@interface Root
@end
@implementation Root
@end

#ifdef REDUNDANCY
// REDUNDANCY1-NOT: _OBJC_CLASS_PROTOCOLS_$_Implementer{{.*}}_OBJC_PROTOCOL_$_B
// REDUNDANCY2:     _OBJC_CLASS_PROTOCOLS_$_Implementer{{.*}}_OBJC_PROTOCOL_$_C{{.*}}_OBJC_PROTOCOL_$_A
@protocol C
@end
@protocol B <C>
@end
@protocol A <B>
@end
__attribute__((objc_non_runtime_protocol)) @protocol Alpha<A>
@end
__attribute__((objc_non_runtime_protocol)) @protocol Beta<B>
@end
@interface Implementer : Root <Alpha, Beta, C>
@end
@implementation Implementer
@end
#endif

#ifdef BASE
// Confirm that we're not emitting protocol information for the
// NONFRAGILE-NOT: OBJC_CLASS_NAME{{.*}}NonRuntimeProtocol
// NONFRAGILE-NOT: _OBJC_$_PROTOCOL_INSTANCE_METHODS_NonRuntimeProtocol
// NONFRAGILE-NOT: _OBJC_$_PROTOCOL_CLASS_METHODS_NonRuntimeProtocol
// NONFRAGILE-NOT: _OBJC_PROTOCOL_$_NonRuntimeProtocol
// NONFRAGILE-NOT: _OBJC_LABEL_PROTOCOL_$_NonRuntimeProtocol
// NONFRAGILE-NOT: _OBJC_CLASS_PROTOCOLS_$_NonRuntimeImplementer
// FRAGILE-NOT: OBJC_CLASS_NAME_.{{.*}}"Runtime\00"
// FRAGILE-NOT: OBJC_PROTOCOL_NonRuntime
// FRAGILE-NOT: OBJC_PROTOCOLS_NonRuntimeImplementer
// GNU-NOT: private unnamed_addr constant {{.*}} c"NonRuntimeProtocol\00"
// GNU-NOT: @.objc_protocol {{.*}}
// GNU2-NOT: private unnamed_addr constant {{.*}} c"NonRuntimeProtocol\00"
// GNU2-NOT: @.objc_protocol {{.*}}
__attribute__((objc_non_runtime_protocol))
@protocol NonRuntimeProtocol
- (void)doThing;
+ (void)doClassThing;
@end
// NONFRAGILE: @"_OBJC_METACLASS_RO_$_NonRuntimeImplementer" {{.*}} %struct._objc_protocol_list* null
// NONFRAGILE: @"_OBJC_CLASS_RO_$_NonRuntimeImplementer" {{.*}} %struct._objc_protocol_list* null
@interface NonRuntimeImplementer : Root <NonRuntimeProtocol>
- (void)doThing;
+ (void)doClassThing;
@end

@implementation NonRuntimeImplementer
- (void)doThing {
}
+ (void)doClassThing {
}
@end
#endif

#ifdef PROTOEXPR
__attribute__((objc_non_runtime_protocol))
@protocol NonRuntimeProtocol
@end
void use() {
  // PROTOEXPR: cannot use a protocol declared 'objc_non_runtime_protocol' in a @protocol expression
  Protocol *p = @protocol(NonRuntimeProtocol);
}
#endif

#ifdef INHERITANCE
// Confirm that we only emit references to the non-runtime protocols and
// properly walk the DAG to find the right protocols.
// INHERITANCE: OBJC_PROTOCOL_$_R2{{.*}}
// INHERITANCE: OBJC_PROTOCOL_$_R3{{.*}}
// INHERITANCE: @"_OBJC_CLASS_PROTOCOLS_$_Implementer" {{.*}}_OBJC_PROTOCOL_$_R2{{.*}}_OBJC_PROTOCOL_$_R3

// FRAGILEINHERITANCE: OBJC_PROTOCOL_R2
// FRAGILEINHERITANCE: OBJC_PROTOCOL_R3
// FRAGILEINHERITANCE: OBJC_CLASS_PROTOCOLS_Implementer{{.*}}OBJC_PROTOCOL_R2{{.*}}OBJC_PROTOCOL_R3

// GNUINHERITANCE-DAG: @[[Proto1:[0-9]]]{{.*}}c"R1\00"
// GNUINHERITANCE-DAG: [[P1Name:@.objc_protocol.[0-9]*]]{{.*}}@[[Proto1]]
// GNUINHERITANCE-DAG: @[[Proto2:[0-9]]]{{.*}}c"R2\00"
// GNUINHERITANCE-DAG: [[P2Name:@.objc_protocol.[0-9]+]]{{.*}}@[[Proto2]]
// GNUINHERITANCE-DAG: @[[Proto3:[0-9]]]{{.*}}c"R3\00"
// GNUINHERITANCE-DAG: [[P3Name:@.objc_protocol.[0-9]+]]{{.*}}@[[Proto3]]
// GNUINHERITANCE-DAG: @.objc_protocol_list{{.*}}
// GNUINHERITANCE: @.objc_protocol_list{{.*}}[[Proto3]]{{.*}}[[Proto2]]

// GNU2INHERITANCE-DAG: @[[Proto1:[0-9]]]{{.*}}c"R1\00"
// GNU2INHERITANCE-DAG: _OBJC_PROTOCOL_R1{{.*}}@[[Proto1]]
// GNU2INHERITANCE-DAG: @[[Proto2:[0-9]]]{{.*}}c"R2\00"
// GNU2INHERITANCE-DAG: _OBJC_PROTOCOL_R2{{.*}}@[[Proto2]]
// GNU2INHERITANCE-DAG: @[[Proto3:[0-9]]]{{.*}}c"R3\00"
// GNU2INHERITANCE-DAG: _OBJC_PROTOCOL_R3{{.*}}@[[Proto3]]
// GNU2INHERITANCE: @.objc_protocol_list{{.*}}_OBJC_PROTOCOL_R2{{.*}}_OBJC_PROTOCOL_R3
@protocol R1
@end
@protocol R2
@end
@protocol R3 <R1>
@end
__attribute__((objc_non_runtime_protocol)) @protocol N3
@end
__attribute__((objc_non_runtime_protocol)) @protocol N1<R3, R2, N3>
@end
__attribute__((objc_non_runtime_protocol)) @protocol N2<N1, R2>
@end
@interface Implementer : Root <N2, R2>
@end
@implementation Implementer
@end
#endif
