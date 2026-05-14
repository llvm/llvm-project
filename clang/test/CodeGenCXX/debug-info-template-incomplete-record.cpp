// RUN: %clang_cc1 -std=c++20 -debug-info-kind=standalone -fvisibility=hidden \
// RUN:   -O2 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s
//
// Verify that we don't crash when generating debug info for a template class
// that is still being defined when its forward declaration is needed. The bug
// was that getTypeIdentifier() called getVTableLinkage() on a record that was
// only "being defined" (not yet complete), which triggered premature
// ASTRecordLayout computation and cached a layout with zero fields. Later,
// CollectRecordFields used the stale cached layout and crashed on an
// out-of-bounds field offset access.

class a;
template<class>
class b;
class c
{
public:
   using d = a;
};
template<class>
class f;
template<class e>
class h
{
   using j = typename e::j;
   j k();
};
class l;
class m
{
   using i = f<l>;
   i n();
};
template<class>
class f
{
public:
   using e = l;
   using o = e;
   using d = b<e>;
   d *p;
};
class L;
class q
{
public:
   using j = L;
};
class r
{
   ~r();
   h<q> s;
};
r::~r() = default;
class l
{
public:
   using j = m;
   using g = c;
};
class J
{
   h<l> t;
};
class a
{
public:
   virtual ~a();
};
template<class e>
class b : e::g::d
{
   using u = typename f<e>::o;
   int v;
};
class L : J
{
};
extern template class b<l>;

// Just verify we produce debug info for b<l> with its field v.
// CHECK: !DICompositeType(tag: DW_TAG_class_type, name: "b<l>"
// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "v"
