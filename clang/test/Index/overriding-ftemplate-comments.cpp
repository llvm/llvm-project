// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng %s > %t/out
// RUN: FileCheck %s < %t/out
// Test to search overridden methods for documentation when overriding method has none. rdar://12378793

// Ensure that XML we generate is not invalid.
// RUN: FileCheck %s -check-prefix=WRONG < %t/out
// WRONG-NOT: CommentXMLInvalid

/// \tparam
/// \param AAA Blah blah
template<typename T>
void comment_to_html_conversion_17(T AAA);

template<typename T>
void comment_to_html_conversion_17(T PPP);

/// \tparam BBB Bbb
/// \tparam AAA Aaa
template<typename AAA, typename BBB>
void comment_to_html_conversion_19(AAA aaa, BBB bbb);

template<typename PPP, typename QQQ>
void comment_to_html_conversion_19(PPP aaa, QQQ bbb);

/// \tparam BBB Bbb
/// \tparam UUU Zzz
/// \tparam CCC Ccc
/// \tparam AAA Aaa
template<typename AAA, typename BBB, int CCC>
void comment_to_html_conversion_20(AAA aaa, BBB bbb);

template<typename PPP, typename QQQ, int RRR>
void comment_to_html_conversion_20(PPP aaa, QQQ bbb);

/// \tparam AAA Aaa
/// \tparam BBB Bbb
/// \tparam CCC Ccc
/// \tparam DDD Ddd
template<template<template<typename CCC> class DDD, class BBB> class AAA>
void comment_to_html_conversion_21();

template<template<template<typename RRR> class SSS, class QQQ> class PPP>
void comment_to_html_conversion_21();

/// \tparam C1 Ccc 1
/// \tparam AAA Zzz
/// \tparam C2 Ccc 2
/// \tparam C3 Ccc 3
/// \tparam C4 Ccc 4
/// \tparam BBB Bbb
template<class C1, template<class C2, template<class C3, class C4> class BBB> class AAA>
void comment_to_html_conversion_22();


template<class CCC1, template<class CCC2, template<class CCC3, class CCC4> class QQQ> class PPP>
void comment_to_html_conversion_22();

// CHECK: FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}overriding-ftemplate-comments.cpp" line="14" column="6"><Name>comment_to_html_conversion_17</Name><USR>c:@FT@&gt;1#Tcomment_to_html_conversion_17#t0.0#</USR><Declaration>template &lt;typename T&gt; void comment_to_html_conversion_17(T AAA)</Declaration><Parameters><Parameter><Name>AAA</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Blah blah</Para></Discussion></Parameter></Parameters></Function>]

// CHECK: FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}overriding-ftemplate-comments.cpp" line="17" column="6"><Name>comment_to_html_conversion_17</Name><USR>c:@FT@&gt;1#Tcomment_to_html_conversion_17#t0.0#</USR><Declaration>template &lt;typename T&gt; void comment_to_html_conversion_17(T PPP)</Declaration><Parameters><Parameter><Name>PPP</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Blah blah</Para></Discussion></Parameter></Parameters></Function>]

// CHECK: FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}overriding-ftemplate-comments.cpp" line="22" column="6"><Name>comment_to_html_conversion_19</Name><USR>c:@FT@&gt;2#T#Tcomment_to_html_conversion_19#t0.0#t0.1#</USR><Declaration>template &lt;typename AAA, typename BBB&gt; void comment_to_html_conversion_19(AAA aaa, BBB bbb)</Declaration><TemplateParameters><Parameter><Name>AAA</Name><Index>0</Index><Discussion><Para> Aaa</Para></Discussion></Parameter><Parameter><Name>BBB</Name><Index>1</Index><Discussion><Para> Bbb </Para></Discussion></Parameter></TemplateParameters></Function>]

// CHECK: FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}overriding-ftemplate-comments.cpp" line="25" column="6"><Name>comment_to_html_conversion_19</Name><USR>c:@FT@&gt;2#T#Tcomment_to_html_conversion_19#t0.0#t0.1#</USR><Declaration>template &lt;typename PPP, typename QQQ&gt; void comment_to_html_conversion_19(PPP aaa, QQQ bbb)</Declaration><TemplateParameters><Parameter><Name>PPP</Name><Index>0</Index><Discussion><Para> Aaa</Para></Discussion></Parameter><Parameter><Name>QQQ</Name><Index>1</Index><Discussion><Para> Bbb </Para></Discussion></Parameter></TemplateParameters></Function>]

// CHECK: FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}overriding-ftemplate-comments.cpp" line="32" column="6"><Name>comment_to_html_conversion_20</Name><USR>c:@FT@&gt;3#T#T#NIcomment_to_html_conversion_20#t0.0#t0.1#</USR><Declaration>template &lt;typename AAA, typename BBB, int CCC&gt; void comment_to_html_conversion_20(AAA aaa, BBB bbb)</Declaration><TemplateParameters><Parameter><Name>AAA</Name><Index>0</Index><Discussion><Para> Aaa</Para></Discussion></Parameter><Parameter><Name>BBB</Name><Index>1</Index><Discussion><Para> Bbb </Para></Discussion></Parameter><Parameter><Name>CCC</Name><Index>2</Index><Discussion><Para> Ccc </Para></Discussion></Parameter><Parameter><Name>UUU</Name><Discussion><Para> Zzz </Para></Discussion></Parameter></TemplateParameters></Function>]

// CHECK: FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}overriding-ftemplate-comments.cpp" line="35" column="6"><Name>comment_to_html_conversion_20</Name><USR>c:@FT@&gt;3#T#T#NIcomment_to_html_conversion_20#t0.0#t0.1#</USR><Declaration>template &lt;typename PPP, typename QQQ, int RRR&gt; void comment_to_html_conversion_20(PPP aaa, QQQ bbb)</Declaration><TemplateParameters><Parameter><Name>PPP</Name><Index>0</Index><Discussion><Para> Aaa</Para></Discussion></Parameter><Parameter><Name>QQQ</Name><Index>1</Index><Discussion><Para> Bbb </Para></Discussion></Parameter><Parameter><Name>RRR</Name><Index>2</Index><Discussion><Para> Ccc </Para></Discussion></Parameter><Parameter><Name>UUU</Name><Discussion><Para> Zzz </Para></Discussion></Parameter></TemplateParameters></Function>]

// CHECK: FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}overriding-ftemplate-comments.cpp" line="42" column="6"><Name>comment_to_html_conversion_21</Name><USR>c:@FT@&gt;1#t&gt;2#t&gt;1#T#Tcomment_to_html_conversion_21#</USR><Declaration>template &lt;template &lt;template &lt;typename CCC&gt; class DDD, class BBB&gt; class AAA&gt; void comment_to_html_conversion_21()</Declaration><TemplateParameters><Parameter><Name>AAA</Name><Index>0</Index><Discussion><Para> Aaa </Para></Discussion></Parameter><Parameter><Name>BBB</Name><Discussion><Para> Bbb </Para></Discussion></Parameter><Parameter><Name>CCC</Name><Discussion><Para> Ccc </Para></Discussion></Parameter><Parameter><Name>DDD</Name><Discussion><Para> Ddd</Para></Discussion></Parameter></TemplateParameters></Function>]

// CHECK: FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}overriding-ftemplate-comments.cpp" line="45" column="6"><Name>comment_to_html_conversion_21</Name><USR>c:@FT@&gt;1#t&gt;2#t&gt;1#T#Tcomment_to_html_conversion_21#</USR><Declaration>template &lt;template &lt;template &lt;typename RRR&gt; class SSS, class QQQ&gt; class PPP&gt; void comment_to_html_conversion_21()</Declaration><TemplateParameters><Parameter><Name>PPP</Name><Index>0</Index><Discussion><Para> Aaa </Para></Discussion></Parameter><Parameter><Name>QQQ</Name><Discussion><Para> Bbb </Para></Discussion></Parameter><Parameter><Name>RRR</Name><Discussion><Para> Ccc </Para></Discussion></Parameter><Parameter><Name>SSS</Name><Discussion><Para> Ddd</Para></Discussion></Parameter></TemplateParameters></Function>]

// CHECK: FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}overriding-ftemplate-comments.cpp" line="54" column="6"><Name>comment_to_html_conversion_22</Name><USR>c:@FT@&gt;2#T#t&gt;2#T#t&gt;2#T#Tcomment_to_html_conversion_22#</USR><Declaration>template &lt;class C1, template &lt;class C2, template &lt;class C3, class C4&gt; class BBB&gt; class AAA&gt; void comment_to_html_conversion_22()</Declaration><TemplateParameters><Parameter><Name>C1</Name><Index>0</Index><Discussion><Para> Ccc 1 </Para></Discussion></Parameter><Parameter><Name>AAA</Name><Index>1</Index><Discussion><Para> Zzz </Para></Discussion></Parameter><Parameter><Name>C2</Name><Discussion><Para> Ccc 2 </Para></Discussion></Parameter><Parameter><Name>C3</Name><Discussion><Para> Ccc 3 </Para></Discussion></Parameter><Parameter><Name>C4</Name><Discussion><Para> Ccc 4 </Para></Discussion></Parameter><Parameter><Name>BBB</Name><Discussion><Para> Bbb</Para></Discussion></Parameter></TemplateParameters></Function>]

// CHECK: FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}overriding-ftemplate-comments.cpp" line="58" column="6"><Name>comment_to_html_conversion_22</Name><USR>c:@FT@&gt;2#T#t&gt;2#T#t&gt;2#T#Tcomment_to_html_conversion_22#</USR><Declaration>template &lt;class CCC1, template &lt;class CCC2, template &lt;class CCC3, class CCC4&gt; class QQQ&gt; class PPP&gt; void comment_to_html_conversion_22()</Declaration><TemplateParameters><Parameter><Name>CCC1</Name><Index>0</Index><Discussion><Para> Ccc 1 </Para></Discussion></Parameter><Parameter><Name>PPP</Name><Index>1</Index><Discussion><Para> Zzz </Para></Discussion></Parameter><Parameter><Name>CCC2</Name><Discussion><Para> Ccc 2 </Para></Discussion></Parameter><Parameter><Name>CCC3</Name><Discussion><Para> Ccc 3 </Para></Discussion></Parameter><Parameter><Name>CCC4</Name><Discussion><Para> Ccc 4 </Para></Discussion></Parameter><Parameter><Name>QQQ</Name><Discussion><Para> Bbb</Para></Discussion></Parameter></TemplateParameters></Function>]

