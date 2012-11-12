// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -fobjc-gc -emit-llvm -o %t %s
// RUN: grep -F '@objc_assign_global' %t  | count 26

@class NSObject;
typedef const struct __CFDictionary * CFDictionaryRef;
typedef struct {
  id  element;
  id elementArray[10];
  __strong CFDictionaryRef cfElement;
  __strong CFDictionaryRef cfElementArray[10];
} struct_with_ids_t;


// assignments to these should generate objc_assign_global
@interface A
@end

typedef struct s0 {
  A *a[4];
} T;

T g0;

extern id FileExternID;
static id FileStaticID;
id GlobalId;
id GlobalArray[20];
NSObject *GlobalObject;
NSObject *GlobalObjectArray[20];
__strong CFDictionaryRef Gdict;
__strong CFDictionaryRef Gdictarray[10];
struct_with_ids_t GlobalStruct;
struct_with_ids_t GlobalStructArray[10];

#define ASSIGNTEST(expr, global) expr = rhs
void *rhs = 0;

int main() {
  static id staticGlobalId;
  static id staticGlobalArray[20];
  static NSObject *staticGlobalObject;
  static NSObject *staticGlobalObjectArray[20];
  static __strong CFDictionaryRef staticGdict;
  static __strong CFDictionaryRef staticGdictarray[10];
  static struct_with_ids_t staticGlobalStruct;
  static struct_with_ids_t staticGlobalStructArray[10];
  extern id ExID;
  id localID;

  ASSIGNTEST(GlobalId, GlobalAssigns);                          // objc_assign_global
  ASSIGNTEST(GlobalArray[0], GlobalAssigns);                    // objc_assign_global
  ASSIGNTEST(GlobalObject, GlobalAssigns);                      // objc_assign_global
  ASSIGNTEST(GlobalObjectArray[0], GlobalAssigns);              // objc_assign_global
  ASSIGNTEST(Gdict, GlobalAssigns);                             // objc_assign_global
  ASSIGNTEST(Gdictarray[1], GlobalAssigns);                     // objc_assign_global

  ASSIGNTEST(GlobalStruct.element, GlobalAssigns);              // objc_assign_global
  ASSIGNTEST(GlobalStruct.elementArray[0], GlobalAssigns);      // objc_assign_global
  ASSIGNTEST(GlobalStruct.cfElement, GlobalAssigns);            // objc_assign_global
  ASSIGNTEST(GlobalStruct.cfElementArray[0], GlobalAssigns);    // objc_assign_global

  ASSIGNTEST(staticGlobalId, GlobalAssigns);                    // objc_assign_global
  ASSIGNTEST(staticGlobalArray[0], GlobalAssigns);              // objc_assign_global
  ASSIGNTEST(staticGlobalObject, GlobalAssigns);                // objc_assign_global
  ASSIGNTEST(staticGlobalObjectArray[0], GlobalAssigns);        // objc_assign_global
  ASSIGNTEST(staticGdict, GlobalAssigns);                       // objc_assign_global
  ASSIGNTEST(staticGdictarray[1], GlobalAssigns);               // objc_assign_global

  ASSIGNTEST(staticGlobalStruct.element, GlobalAssigns);                // objc_assign_global
  ASSIGNTEST(staticGlobalStruct.elementArray[0], GlobalAssigns);        // objc_assign_global
  ASSIGNTEST(staticGlobalStruct.cfElement, GlobalAssigns);              // objc_assign_global
  ASSIGNTEST(staticGlobalStruct.cfElementArray[0], GlobalAssigns);      // objc_assign_global

  ExID = 0;
  localID = 0;
  FileStaticID = 0;
  FileExternID=0;
  g0.a[0] = 0;
  ((T*) &g0)->a[0] = 0;
}
