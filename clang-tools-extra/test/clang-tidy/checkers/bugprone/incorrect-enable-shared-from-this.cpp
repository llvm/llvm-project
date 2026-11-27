// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-incorrect-enable-shared-from-this %t

// NOLINTBEGIN
namespace std {
    template <typename T> class enable_shared_from_this {};
} //namespace std
// NOLINTEND

class BadClassExample : std::enable_shared_from_this<BadClassExample> {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'BadClassExample' is not publicly inheriting from 'std::enable_shared_from_this', which will cause unintended behaviour when using 'shared_from_this'; make the inheritance public [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: class BadClassExample : public std::enable_shared_from_this<BadClassExample> {};

class BadClass2Example : private std::enable_shared_from_this<BadClass2Example> {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'BadClass2Example' is not publicly inheriting from 'std::enable_shared_from_this', which will cause unintended behaviour when using 'shared_from_this'; make the inheritance public [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: class BadClass2Example : public std::enable_shared_from_this<BadClass2Example> {};

struct BadStructExample : private std::enable_shared_from_this<BadStructExample> {};
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: 'BadStructExample' is not publicly inheriting from 'std::enable_shared_from_this', which will cause unintended behaviour when using 'shared_from_this'; make the inheritance public [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: struct BadStructExample : public std::enable_shared_from_this<BadStructExample> {};

class GoodClassExample : public std::enable_shared_from_this<GoodClassExample> {};

struct GoodStructExample : public std::enable_shared_from_this<GoodStructExample> {};

struct GoodStruct2Example : std::enable_shared_from_this<GoodStruct2Example> {};

class dummy_class1 {};
class dummy_class2 {};

class BadMultiClassExample : std::enable_shared_from_this<BadMultiClassExample>, dummy_class1 {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'BadMultiClassExample' is not publicly inheriting from 'std::enable_shared_from_this', which will cause unintended behaviour when using 'shared_from_this'; make the inheritance public [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: class BadMultiClassExample : public std::enable_shared_from_this<BadMultiClassExample>, dummy_class1 {};

class BadMultiClass2Example : dummy_class1, std::enable_shared_from_this<BadMultiClass2Example>, dummy_class2 {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'BadMultiClass2Example' is not publicly inheriting from 'std::enable_shared_from_this', which will cause unintended behaviour when using 'shared_from_this'; make the inheritance public [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: class BadMultiClass2Example : dummy_class1, public std::enable_shared_from_this<BadMultiClass2Example>, dummy_class2 {};

class BadMultiClass3Example : dummy_class1, dummy_class2, std::enable_shared_from_this<BadMultiClass3Example> {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'BadMultiClass3Example' is not publicly inheriting from 'std::enable_shared_from_this', which will cause unintended behaviour when using 'shared_from_this'; make the inheritance public [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: class BadMultiClass3Example : dummy_class1, dummy_class2, public std::enable_shared_from_this<BadMultiClass3Example> {};

class ClassBase : public std::enable_shared_from_this<ClassBase> {};
class PrivateInheritClassBase : private ClassBase {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'PrivateInheritClassBase' is not publicly inheriting from 'ClassBase' which inherits from 'std::enable_shared_from_this', which will cause unintended behaviour when using 'shared_from_this'; make the inheritance public [bugprone-incorrect-enable-shared-from-this]

class DefaultInheritClassBase : ClassBase {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'DefaultInheritClassBase' is not publicly inheriting from 'ClassBase' which inherits from 'std::enable_shared_from_this', which will cause unintended behaviour when using 'shared_from_this'; make the inheritance public [bugprone-incorrect-enable-shared-from-this]

class PublicInheritClassBase : public ClassBase {};

struct StructBase : public std::enable_shared_from_this<StructBase> {};
struct PrivateInheritStructBase : private StructBase {};
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: 'PrivateInheritStructBase' is not publicly inheriting from 'StructBase' which inherits from 'std::enable_shared_from_this', which will cause unintended behaviour when using 'shared_from_this'; make the inheritance public [bugprone-incorrect-enable-shared-from-this]

struct DefaultInheritStructBase : StructBase {};

struct PublicInheritStructBase : StructBase {};

//alias the template itself
template <typename T> using esft_template = std::enable_shared_from_this<T>;

class PrivateAliasTemplateClassBase : private esft_template<PrivateAliasTemplateClassBase> {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'PrivateAliasTemplateClassBase' is not publicly inheriting from 'std::enable_shared_from_this', which will cause unintended behaviour when using 'shared_from_this'; make the inheritance public [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: class PrivateAliasTemplateClassBase : public esft_template<PrivateAliasTemplateClassBase> {};

class DefaultAliasTemplateClassBase : esft_template<DefaultAliasTemplateClassBase> {}; 
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'DefaultAliasTemplateClassBase' is not publicly inheriting from 'std::enable_shared_from_this', which will cause unintended behaviour when using 'shared_from_this'; make the inheritance public [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: class DefaultAliasTemplateClassBase : public esft_template<DefaultAliasTemplateClassBase> {};

class PublicAliasTemplateClassBase : public esft_template<PublicAliasTemplateClassBase> {}; 

struct PrivateAliasTemplateStructBase : private esft_template<PrivateAliasTemplateStructBase> {}; 
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: 'PrivateAliasTemplateStructBase' is not publicly inheriting from 'std::enable_shared_from_this', which will cause unintended behaviour when using 'shared_from_this'; make the inheritance public [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: struct PrivateAliasTemplateStructBase : public esft_template<PrivateAliasTemplateStructBase> {};

struct DefaultAliasTemplateStructBase : esft_template<DefaultAliasTemplateStructBase> {}; 

struct PublicAliasTemplateStructBase : public esft_template<PublicAliasTemplateStructBase> {}; 

//alias with specific instance
using esft = std::enable_shared_from_this<ClassBase>;
class PrivateAliasClassBase : private esft {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'PrivateAliasClassBase' is not publicly inheriting from 'std::enable_shared_from_this', which will cause unintended behaviour when using 'shared_from_this'; make the inheritance public [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: class PrivateAliasClassBase : public esft {};

class DefaultAliasClassBase : esft {}; 
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'DefaultAliasClassBase' is not publicly inheriting from 'std::enable_shared_from_this', which will cause unintended behaviour when using 'shared_from_this'; make the inheritance public [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: class DefaultAliasClassBase : public esft {};

class PublicAliasClassBase : public esft {}; 

struct PrivateAliasStructBase : private esft {}; 
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: 'PrivateAliasStructBase' is not publicly inheriting from 'std::enable_shared_from_this', which will cause unintended behaviour when using 'shared_from_this'; make the inheritance public [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: struct PrivateAliasStructBase : public esft {};

struct DefaultAliasStructBase : esft {}; 

struct PublicAliasStructBase : public esft {}; 

//we can only typedef a specific instance of the template
typedef std::enable_shared_from_this<ClassBase> EnableSharedFromThis;
class PrivateTypedefClassBase : private EnableSharedFromThis {}; 
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'PrivateTypedefClassBase' is not publicly inheriting from 'std::enable_shared_from_this', which will cause unintended behaviour when using 'shared_from_this'; make the inheritance public [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: class PrivateTypedefClassBase : public EnableSharedFromThis {};

class DefaultTypedefClassBase : EnableSharedFromThis {}; 
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'DefaultTypedefClassBase' is not publicly inheriting from 'std::enable_shared_from_this', which will cause unintended behaviour when using 'shared_from_this'; make the inheritance public [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: class DefaultTypedefClassBase : public EnableSharedFromThis {};

class PublicTypedefClassBase : public EnableSharedFromThis {}; 

struct PrivateTypedefStructBase : private EnableSharedFromThis {}; 
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: 'PrivateTypedefStructBase' is not publicly inheriting from 'std::enable_shared_from_this', which will cause unintended behaviour when using 'shared_from_this'; make the inheritance public [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: struct PrivateTypedefStructBase : public EnableSharedFromThis {};

struct DefaultTypedefStructBase : EnableSharedFromThis {}; 

struct PublicTypedefStructBase : public EnableSharedFromThis {}; 

#define PRIVATE_ESFT_CLASS(ClassName) \
   class ClassName: private std::enable_shared_from_this<ClassName> { \
   };

PRIVATE_ESFT_CLASS(PrivateEsftClass);
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: 'PrivateEsftClass' is not publicly inheriting from 'std::enable_shared_from_this', which will cause unintended behaviour when using 'shared_from_this'; make the inheritance public [bugprone-incorrect-enable-shared-from-this]

#define DEFAULT_ESFT_CLASS(ClassName) \
   class ClassName: std::enable_shared_from_this<ClassName> { \
   };

DEFAULT_ESFT_CLASS(DefaultEsftClass);
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: 'DefaultEsftClass' is not publicly inheriting from 'std::enable_shared_from_this', which will cause unintended behaviour when using 'shared_from_this'; make the inheritance public [bugprone-incorrect-enable-shared-from-this]

#define PUBLIC_ESFT_CLASS(ClassName) \
   class ClassName: public std::enable_shared_from_this<ClassName> { \
   };

PUBLIC_ESFT_CLASS(PublicEsftClass);

#define PRIVATE_ESFT_STRUCT(StructName) \
   struct StructName: private std::enable_shared_from_this<StructName> { \
   };

PRIVATE_ESFT_STRUCT(PrivateEsftStruct);
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: 'PrivateEsftStruct' is not publicly inheriting from 'std::enable_shared_from_this', which will cause unintended behaviour when using 'shared_from_this'; make the inheritance public [bugprone-incorrect-enable-shared-from-this]

#define DEFAULT_ESFT_STRUCT(StructName) \
   struct StructName: std::enable_shared_from_this<StructName> { \
   };

DEFAULT_ESFT_STRUCT(DefaultEsftStruct);

#define PUBLIC_ESFT_STRUCT(StructName) \
   struct StructName: std::enable_shared_from_this<StructName> { \
   };

PUBLIC_ESFT_STRUCT(PublicEsftStruct);

struct A : std::enable_shared_from_this<A> {};
#define MACRO_A A

class B : MACRO_A {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'B' is not publicly inheriting from 'A' which inherits from 'std::enable_shared_from_this', which will cause unintended behaviour when using 'shared_from_this'; make the inheritance public [bugprone-incorrect-enable-shared-from-this]

class C : private MACRO_A {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'C' is not publicly inheriting from 'A' which inherits from 'std::enable_shared_from_this', which will cause unintended behaviour when using 'shared_from_this'; make the inheritance public [bugprone-incorrect-enable-shared-from-this]

class D : public MACRO_A {};

#define MACRO_PARAM(CLASS) std::enable_shared_from_this<CLASS>

class E : MACRO_PARAM(E) {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'E' is not publicly inheriting from 'std::enable_shared_from_this', which will cause unintended behaviour when using 'shared_from_this'; make the inheritance public [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: class E : public MACRO_PARAM(E) {};

class F : private MACRO_PARAM(F) {};
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'F' is not publicly inheriting from 'std::enable_shared_from_this', which will cause unintended behaviour when using 'shared_from_this'; make the inheritance public [bugprone-incorrect-enable-shared-from-this]
// CHECK-FIXES: class F : public MACRO_PARAM(F) {};

class G : public MACRO_PARAM(G) {};
