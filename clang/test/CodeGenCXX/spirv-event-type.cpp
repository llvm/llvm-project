// REQUIRES: spirv-registered-target
// RUN: %clang_cc1 -triple spirv64 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spirv64 -emit-llvm -debug-info-kind=limited -o - %s | FileCheck %s --check-prefix=DEBUG

// Verify that __spirv_event_t lowers to the target("spirv.Event") extension
// type and that it mangles as a vendor extended type.

// Verify debug information generated for spirv.Event.

namespace std { class type_info; }

__spirv_event_t getEvent();
void consumeEvent(__spirv_event_t e);

const std::type_info &ti = typeid(__spirv_event_t);
// CHECK: @_ZTIu15__spirv_event_t = {{.*}}constant { {{.*}} } { {{.*}}@_ZTVN10__cxxabiv123__fundamental_type_infoE{{.*}}, {{.*}}@_ZTSu15__spirv_event_t {{.*}}}
// CHECK: @_ZTSu15__spirv_event_t = {{.*}}constant [19 x i8] c"u15__spirv_event_t\00"

__spirv_event_t __attribute__((address_space(7))) as_event;
// CHECK: @as_event = addrspace(7) global target("spirv.Event") zeroinitializer

void test(__spirv_event_t e) {
// CHECK: define spir_func void @_Z4testu15__spirv_event_t(target("spirv.Event") %e)
  __spirv_event_t events[4];
// CHECK: %events = alloca [4 x target("spirv.Event")]
  events[0] = getEvent();
// CHECK: %[[GEP:.*]] = getelementptr inbounds [4 x target("spirv.Event")], ptr %events, i64 0, i64 0
// CHECK: store target("spirv.Event") {{.*}}, ptr %[[GEP]]
  __spirv_event_t copyConstructed = e;
// CHECK: %[[CC:.*]] = load target("spirv.Event"), ptr %e.addr
// CHECK: store target("spirv.Event") %[[CC]], ptr %copyConstructed
  __spirv_event_t moveConstructed = getEvent();
// CHECK: %[[MC:.*]] = call spir_func target("spirv.Event") @_Z8getEventv()
// CHECK: store target("spirv.Event") %[[MC]], ptr %moveConstructed
  copyConstructed = e;
// CHECK: %[[CA:.*]] = load target("spirv.Event"), ptr %e.addr
// CHECK: store target("spirv.Event") %[[CA]], ptr %copyConstructed
  moveConstructed = getEvent();
// CHECK: %[[MA:.*]] = call spir_func target("spirv.Event") @_Z8getEventv()
// CHECK: store target("spirv.Event") %[[MA]], ptr %moveConstructed
  consumeEvent(copyConstructed);
// CHECK: %[[LD:.*]] = load target("spirv.Event"), ptr %copyConstructed
// CHECK: call spir_func void @_Z12consumeEventu15__spirv_event_t(target("spirv.Event") %[[LD]])
}

// CHECK: declare spir_func target("spirv.Event") @_Z8getEventv()
// CHECK: declare spir_func void @_Z12consumeEventu15__spirv_event_t(target("spirv.Event"))

// DEBUG: ![[TD:[0-9]+]] = !DIDerivedType(tag: DW_TAG_typedef, name: "__spirv_event_t", {{.*}} baseType: ![[PTR:[0-9]+]])
// DEBUG: ![[PTR]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[BT:[0-9]+]]
// DEBUG: ![[BT]] = !DICompositeType(tag: DW_TAG_structure_type, name: "__spirv_event_t", {{.*}} flags: DIFlagFwdDecl)
// DEBUG-DAG: !DILocalVariable(name: "e", {{.*}} type: ![[TD]])
// DEBUG-DAG: !DILocalVariable(name: "copyConstructed", {{.*}} type: ![[TD]])
// DEBUG-DAG: !DILocalVariable(name: "moveConstructed", {{.*}} type: ![[TD]])

