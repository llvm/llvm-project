// RUN: %clang_cc1 -std=c++26 -freflection -include %S/reflection_include.h -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++26 -freflection -x c++-header %S/reflection_include.h -emit-pch -o %t
// RUN: %clang_cc1 -std=c++26 -freflection -include-pch %t -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++26 -freflection -fexperimental-new-constant-interpreter -include %S/reflection_include.h -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++26 -freflection -fexperimental-new-constant-interpreter -x c++-header %S/reflection_include.h -emit-pch -o %t.new
// RUN: %clang_cc1 -std=c++26 -freflection -fexperimental-new-constant-interpreter -include-pch %t.new -fsyntax-only -verify %s
// expected-no-diagnostics

static_assert(^^int == ^^int);
static_assert(^^int != ^^double);
static_assert(info{} != ^^int);
static_assert(__is_same(decltype(^^int), info));
static_assert(__is_same(decltype(^^double), info));

static_assert(record == ^^Record);
static_assert(alias == ^^Alias);
static_assert(__is_same(Reflected<record>, Reflected<^^Record>));

static_assert(typename_type == ^^int);
static_assert(written_type<WithType>() == ^^int);
static_assert(__is_same(Reflected<typename_type>, Reflected<^^int>));

static_assert(AliasTemplates::defaulted == ^^AliasTemplates::Defaults<int, int>);
static_assert(AliasTemplates::number == ^^AliasTemplates::Numbers<2, 3>);
static_assert(AliasTemplates::packed == ^^AliasTemplates::Pack<int, Record>);
static_assert(AliasTemplates::dependent<float>() == ^^AliasTemplates::Defaults<float>);
static_assert(AliasTemplates::pack<int, Record>() == AliasTemplates::packed);
static_assert(__is_same(decltype(AliasTemplates::defaulted_argument),
                       Reflected<^^AliasTemplates::Defaults<int>>));
static_assert(__is_same(decltype(AliasTemplates::number_argument),
                       Reflected<^^AliasTemplates::Numbers<2>>));
static_assert(__is_same(decltype(AliasTemplates::pack_argument),
                       Reflected<^^AliasTemplates::Pack<int, Record>>));
static_assert(__is_same(decltype(AliasTemplates::ordinary_alias_argument),
                       Reflected<^^Alias>));

static_assert(AliasTemplates::nested ==
              ^^AliasTemplates::Reflection<^^AliasTemplates::Defaults<int>>);
static_assert(__is_same(decltype(AliasTemplates::nested_argument),
                       Reflected<^^AliasTemplates::Reflection<^^AliasTemplates::Defaults<int>>>));

static_assert(identity<^^int>() == ^^int);
static_assert(identity<info{}>() == info{});
static_assert(identity<alias>() == alias);
static_assert(identity<AliasTemplates::defaulted>() == AliasTemplates::defaulted);
