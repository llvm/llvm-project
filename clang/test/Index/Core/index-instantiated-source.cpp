// RUN: c-index-test core -print-source-symbols -- %s -std=c++14 -target x86_64-apple-macosx10.7 | FileCheck %s
// References to declarations in instantiations should be canonicalized:

template<typename T>
class BaseTemplate {
public:
  T baseTemplateFunction();
// CHECK: [[@LINE-1]]:5 | instance-method/C++ | baseTemplateFunction | c:@ST>1#T@BaseTemplate@F@baseTemplateFunction#

  T baseTemplateField;
// CHECK: [[@LINE-1]]:5 | field/C++ | baseTemplateField | c:@ST>1#T@BaseTemplate@FI@baseTemplateField

  struct NestedBaseType { };
// CHECK: [[@LINE-1]]:10 | struct/C | NestedBaseType | c:@ST>1#T@BaseTemplate@S@NestedBaseType |
};

template<typename T, typename S>
class TemplateClass: public BaseTemplate<T> {
public:
  T function() { return T(); }
// CHECK: [[@LINE-1]]:5 | instance-method/C++ | function | c:@ST>2#T#T@TemplateClass@F@function#

  static void staticFunction() { }
// CHECK: [[@LINE-1]]:15 | static-method/C++ | staticFunction | c:@ST>2#T#T@TemplateClass@F@staticFunction#S

  T field;
// CHECK: [[@LINE-1]]:5 | field/C++ | field | c:@ST>2#T#T@TemplateClass@FI@field

  struct NestedType {
// CHECK: [[@LINE-1]]:10 | struct/C++ | NestedType | c:@ST>2#T#T@TemplateClass@S@NestedType |

    T nestedField;
// CHECK: [[@LINE-1]]:7 | field/C++ | nestedField | c:@ST>2#T#T@TemplateClass@S@NestedType@FI@nestedField |

    class SubNestedType {
// CHECK: [[@LINE-1]]:11 | class/C++ | SubNestedType | c:@ST>2#T#T@TemplateClass@S@NestedType@S@SubNestedType |
    public:
      SubNestedType(int);
    };
  };
};

void canonicalizeInstaniationReferences(TemplateClass<int, float> &object) {
  (void)object.function();
// CHECK: [[@LINE-1]]:16 | instance-method/C++ | function | c:@ST>2#T#T@TemplateClass@F@function# | <no-cgname>
  (void)object.field;
// CHECK: [[@LINE-1]]:16 | field/C++ | field | c:@ST>2#T#T@TemplateClass@FI@field | <no-cgname> | Ref,RelCont | rel: 1
  (void)object.baseTemplateFunction();
// CHECK: [[@LINE-1]]:16 | instance-method/C++ | baseTemplateFunction | c:@ST>1#T@BaseTemplate@F@baseTemplateFunction# | <no-cgname>
  (void)object.baseTemplateField;
// CHECK: [[@LINE-1]]:16 | field/C++ | baseTemplateField | c:@ST>1#T@BaseTemplate@FI@baseTemplateField | <no-cgname> | Ref,RelCont | rel: 1

  TemplateClass<int, float>::staticFunction();
// CHECK: [[@LINE-1]]:30 | static-method/C++ | staticFunction | c:@ST>2#T#T@TemplateClass@F@staticFunction#S | <no-cgname

  TemplateClass<int, float>::NestedBaseType nestedBaseType;
// CHECK: [[@LINE-1]]:30 | struct/C | NestedBaseType | c:@ST>1#T@BaseTemplate@S@NestedBaseType |
  TemplateClass<int, float>::NestedType nestedSubType;
// CHECK: [[@LINE-1]]:30 | struct/C++ | NestedType | c:@ST>2#T#T@TemplateClass@S@NestedType |
  (void)nestedSubType.nestedField;
// CHECK: [[@LINE-1]]:23 | field/C++ | nestedField | c:@ST>2#T#T@TemplateClass@S@NestedType@FI@nestedField |

  typedef TemplateClass<int, float> TT;
  TT::NestedType::SubNestedType subNestedType(0);
// CHECK: [[@LINE-1]]:7 | struct/C++ | NestedType | c:@ST>2#T#T@TemplateClass@S@NestedType |
// CHECK: [[@LINE-2]]:19 | class/C++ | SubNestedType | c:@ST>2#T#T@TemplateClass@S@NestedType@S@SubNestedType |
}
