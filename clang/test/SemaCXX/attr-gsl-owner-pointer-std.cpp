// RUN: %clang_cc1 -ast-dump %s | \
// RUN: FileCheck --implicit-check-not OwnerAttr --implicit-check-not PointerAttr %s

// Test attribute inference for types in the standard library.
namespace std {
// Attributes are inferred for a (complete) class.
class any {
  // CHECK: CXXRecordDecl {{.*}} any
  // CHECK: OwnerAttr {{.*}}
};

// Attributes are inferred for instantiations of a complete template.
template <typename T>
class vector {
public:
  class iterator {};
  // CHECK: ClassTemplateDecl {{.*}} vector
  // CHECK: OwnerAttr {{.*}}
  // CHECK: CXXRecordDecl {{.*}} iterator
  // CHECK: PointerAttr {{.*}}
  // CHECK: ClassTemplateSpecializationDecl {{.*}} vector
  // CHECK: TemplateArgument type 'int'
  // CHECK: OwnerAttr
  // CHECK: CXXRecordDecl {{.*}} iterator
  // CHECK: PointerAttr {{.*}}
};
static_assert(sizeof(vector<int>), "");           // Force instantiation.
static_assert(sizeof(vector<int>::iterator), ""); // Force instantiation.

// If std::container::iterator is a using declaration, attributes are inferred
// for the underlying class.
template <typename T>
class __set_iterator {};
// CHECK: ClassTemplateDecl {{.*}} __set_iterator
// CHECK: PointerAttr
// CHECK: ClassTemplateSpecializationDecl {{.*}} __set_iterator
// CHECK: TemplateArgument type 'int'
// CHECK: PointerAttr

template <typename T>
class set {
  // CHECK: ClassTemplateDecl {{.*}} set
  // CHECK: OwnerAttr {{.*}}
  // CHECK: ClassTemplateSpecializationDecl {{.*}} set
  // CHECK: OwnerAttr {{.*}}
public:
  using iterator = __set_iterator<T>;
};
static_assert(sizeof(set<int>::iterator), ""); // Force instantiation.

// If std::container::iterator is a typedef, attributes are inferred for the
// underlying class.
template <typename T>
class __map_iterator {};
// CHECK: ClassTemplateDecl {{.*}} __map_iterator
// CHECK: PointerAttr
// CHECK: ClassTemplateSpecializationDecl {{.*}} __map_iterator
// CHECK: TemplateArgument type 'int'
// CHECK: PointerAttr

template <typename T>
class map {
  // CHECK: ClassTemplateDecl {{.*}} map
  // CHECK: OwnerAttr {{.*}}
  // CHECK: ClassTemplateSpecializationDecl {{.*}} map
  // CHECK: OwnerAttr {{.*}}
public:
  typedef __map_iterator<T> iterator;
};
static_assert(sizeof(map<int>::iterator), ""); // Force instantiation.

// Inline namespaces are ignored when checking if
// the class lives in the std namespace.
inline namespace inlinens {
template <typename T>
class __unordered_map_iterator {};
// CHECK: ClassTemplateDecl {{.*}} __unordered_map_iterator
// CHECK: PointerAttr
// CHECK: ClassTemplateSpecializationDecl {{.*}} __unordered_map_iterator
// CHECK: TemplateArgument type 'int'
// CHECK: PointerAttr

template <typename T>
class unordered_map {
  // CHECK: ClassTemplateDecl {{.*}} unordered_map
  // CHECK: OwnerAttr {{.*}}
  // CHECK: ClassTemplateSpecializationDecl {{.*}} unordered_map
  // CHECK: OwnerAttr {{.*}}
public:
  typedef __unordered_map_iterator<T> iterator;
};
static_assert(sizeof(unordered_map<int>::iterator), ""); // Force instantiation.
} // namespace inlinens

// The iterator typedef is a DependentNameType.
template <typename T>
class __unordered_multimap_iterator {};
// CHECK: ClassTemplateDecl {{.*}} __unordered_multimap_iterator
// CHECK: ClassTemplateSpecializationDecl {{.*}} __unordered_multimap_iterator
// CHECK: TemplateArgument type 'int'
// CHECK: PointerAttr

template <typename T>
class __unordered_multimap_base {
public:
  using iterator = __unordered_multimap_iterator<T>;
};

template <typename T>
class unordered_multimap {
  // CHECK: ClassTemplateDecl {{.*}} unordered_multimap
  // CHECK: OwnerAttr {{.*}}
  // CHECK: ClassTemplateSpecializationDecl {{.*}} unordered_multimap
  // CHECK: OwnerAttr {{.*}}
public:
  using _Mybase = __unordered_multimap_base<T>;
  using iterator = typename _Mybase::iterator;
};
static_assert(sizeof(unordered_multimap<int>::iterator), ""); // Force instantiation.

// The canonical declaration of the iterator template is not its definition.
template <typename T>
class __unordered_multiset_iterator;
// CHECK: ClassTemplateDecl {{.*}} __unordered_multiset_iterator
// CHECK: PointerAttr
// CHECK: ClassTemplateSpecializationDecl {{.*}} __unordered_multiset_iterator
// CHECK: TemplateArgument type 'int'
// CHECK: PointerAttr

template <typename T>
class __unordered_multiset_iterator {
  // CHECK: ClassTemplateDecl {{.*}} prev {{.*}} __unordered_multiset_iterator
  // CHECK: PointerAttr
};

template <typename T>
class unordered_multiset {
  // CHECK: ClassTemplateDecl {{.*}} unordered_multiset
  // CHECK: OwnerAttr {{.*}}
  // CHECK: ClassTemplateSpecializationDecl {{.*}} unordered_multiset
  // CHECK: OwnerAttr {{.*}}
public:
  using iterator = __unordered_multiset_iterator<T>;
};

static_assert(sizeof(unordered_multiset<int>::iterator), ""); // Force instantiation.

// std::list has an implicit gsl::Owner attribute,
// but explicit attributes take precedence.
template <typename T>
class [[gsl::Pointer]] list{};
// CHECK: ClassTemplateDecl {{.*}} list
// CHECK: PointerAttr {{.*}}
// CHECK: ClassTemplateSpecializationDecl {{.*}} list
// CHECK: PointerAttr {{.*}}

static_assert(sizeof(list<int>), ""); // Force instantiation.

// Forward declared template (Owner).
template <
    class CharT,
    class Traits>
class basic_regex;
// CHECK: ClassTemplateDecl {{.*}} basic_regex
// CHECK: OwnerAttr {{.*}}

// Forward declared template (Pointer).
template <class T>
class reference_wrapper;
// CHECK: ClassTemplateDecl {{.*}} reference_wrapper
// CHECK: PointerAttr {{.*}}

class some_unknown_type;
// CHECK: CXXRecordDecl {{.*}} some_unknown_type

} // namespace std

namespace user {
// If a class is not in the std namespace, we don't infer the attributes.
class any {
};
} // namespace user
