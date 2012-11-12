// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng %s > %t/out
// RUN: FileCheck %s < %t/out
// Test to search overridden methods for documentation when overriding method has none. rdar://12378793

// Ensure that XML we generate is not invalid.
// RUN: FileCheck %s -check-prefix=WRONG < %t/out
// WRONG-NOT: CommentXMLInvalid

@protocol P
- (void)METH:(id)PPP;
@end

@interface Root<P>
/**
 * \param[in] AAA ZZZ
 */
- (void)METH:(id)AAA;
@end

@interface Sub : Root
@end

@interface Sub (CAT)
- (void)METH:(id)BBB;
@end

@implementation Sub(CAT)
- (void)METH:(id)III {}
@end

@interface Redec : Root
@end

@interface Redec()
/**
 * \param[in] AAA input value  
 * \param[out] CCC output value is int 
 * \param[in] BBB 2nd input value is double 
 */
- (void)EXT_METH:(id)AAA : (double)BBB : (int)CCC;
@end

@implementation Redec
- (void)EXT_METH:(id)PPP : (double)QQQ : (int)RRR {}
@end

struct Base {
  /// \brief Does something.
  /// \param AAA argument to foo_pure.
  virtual void foo_pure(int AAA) = 0;

  /// \brief Does something.
  /// \param BBB argument to defined virtual.
  virtual void foo_inline(int BBB) {}

  /// \brief Does something.
  /// \param CCC argument to undefined virtual.
  virtual void foo_outofline(int CCC);
};

void Base::foo_outofline(int RRR) {}

struct Derived : public Base {
  virtual void foo_pure(int PPP);

  virtual void foo_inline(int QQQ) {}
};

/// \brief Does something.
/// \param DDD a value.
void foo(int DDD);

void foo(int SSS) {}

/// \brief Does something.
/// \param EEE argument to function decl. 
void foo1(int EEE);

void foo1(int TTT);

/// \brief Documentation
/// \tparam BBB The type, silly.
/// \tparam AAA The type, silly as well.
template<typename AAA, typename BBB>
void foo(AAA, BBB);

template<typename PPP, typename QQQ>
void foo(PPP, QQQ);

// CHECK: FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}overriding-method-comments.mm" line="19" column="1"><Name>METH:</Name><USR>c:objc(cs)Root(im)METH:</USR><Declaration>- (void) METH:(id)AAA</Declaration><Parameters><Parameter><Name>AAA</Name><Index>0</Index><Direction isExplicit="1">in</Direction><Discussion><Para> ZZZ </Para></Discussion></Parameter></Parameters></Function>]

// CHECK: FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}overriding-method-comments.mm" line="26" column="1"><Name>METH:</Name><USR>c:objc(cs)Root(im)METH:</USR><Declaration>- (void) METH:(id)BBB</Declaration><Parameters><Parameter><Name>BBB</Name><Index>0</Index><Direction isExplicit="1">in</Direction><Discussion><Para> ZZZ </Para></Discussion></Parameter></Parameters></Function>]

// CHECK: FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}overriding-method-comments.mm" line="30" column="1"><Name>METH:</Name><USR>c:objc(cs)Root(im)METH:</USR><Declaration>- (void) METH:(id)III</Declaration><Parameters><Parameter><Name>III</Name><Index>0</Index><Direction isExplicit="1">in</Direction><Discussion><Para> ZZZ </Para></Discussion></Parameter></Parameters></Function>]

// CHECK: FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}overriding-method-comments.mm" line="42" column="1"><Name>EXT_METH:::</Name><USR>c:objc(cs)Redec(im)EXT_METH:::</USR><Declaration>- (void) EXT_METH:(id)AAA :(double)BBB :(int)CCC</Declaration><Parameters><Parameter><Name>AAA</Name><Index>0</Index><Direction isExplicit="1">in</Direction><Discussion><Para> input value   </Para></Discussion></Parameter><Parameter><Name>BBB</Name><Index>1</Index><Direction isExplicit="1">in</Direction><Discussion><Para> 2nd input value is double  </Para></Discussion></Parameter><Parameter><Name>CCC</Name><Index>2</Index><Direction isExplicit="1">out</Direction><Discussion><Para> output value is int  </Para></Discussion></Parameter></Parameters></Function>]

// CHECK: FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}overriding-method-comments.mm" line="46" column="1"><Name>EXT_METH:::</Name><USR>c:objc(cs)Redec(im)EXT_METH:::</USR><Declaration>- (void) EXT_METH:(id)PPP :(double)QQQ :(int)RRR</Declaration><Parameters><Parameter><Name>PPP</Name><Index>0</Index><Direction isExplicit="1">in</Direction><Discussion><Para> input value   </Para></Discussion></Parameter><Parameter><Name>QQQ</Name><Index>1</Index><Direction isExplicit="1">in</Direction><Discussion><Para> 2nd input value is double  </Para></Discussion></Parameter><Parameter><Name>RRR</Name><Index>2</Index><Direction isExplicit="1">out</Direction><Discussion><Para> output value is int  </Para></Discussion></Parameter></Parameters></Function>]

// CHECK: FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}overriding-method-comments.mm" line="52" column="16"><Name>foo_pure</Name><USR>c:@S@Base@F@foo_pure#I#</USR><Declaration>virtual void foo_pure(int AAA) = 0</Declaration><Abstract><Para> Does something. </Para></Abstract><Parameters><Parameter><Name>AAA</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> argument to foo_pure.</Para></Discussion></Parameter></Parameters></Function>]

// CHECK: FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}overriding-method-comments.mm" line="56" column="16"><Name>foo_inline</Name><USR>c:@S@Base@F@foo_inline#I#</USR><Declaration>virtual void foo_inline(int BBB)</Declaration><Abstract><Para> Does something. </Para></Abstract><Parameters><Parameter><Name>BBB</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> argument to defined virtual.</Para></Discussion></Parameter></Parameters></Function>]

// CHECK: FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}overriding-method-comments.mm" line="60" column="16"><Name>foo_outofline</Name><USR>c:@S@Base@F@foo_outofline#I#</USR><Declaration>virtual void foo_outofline(int CCC)</Declaration><Abstract><Para> Does something. </Para></Abstract><Parameters><Parameter><Name>CCC</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> argument to undefined virtual.</Para></Discussion></Parameter></Parameters></Function>]

// CHECK: FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}overriding-method-comments.mm" line="63" column="12"><Name>foo_outofline</Name><USR>c:@S@Base@F@foo_outofline#I#</USR><Declaration>void foo_outofline(int RRR)</Declaration><Abstract><Para> Does something. </Para></Abstract><Parameters><Parameter><Name>RRR</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> argument to undefined virtual.</Para></Discussion></Parameter></Parameters></Function>]

// CHECK: FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}overriding-method-comments.mm" line="66" column="16"><Name>foo_pure</Name><USR>c:@S@Base@F@foo_pure#I#</USR><Declaration>virtual void foo_pure(int PPP)</Declaration><Abstract><Para> Does something. </Para></Abstract><Parameters><Parameter><Name>PPP</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> argument to foo_pure.</Para></Discussion></Parameter></Parameters></Function>]

// CHECK: FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}overriding-method-comments.mm" line="68" column="16"><Name>foo_inline</Name><USR>c:@S@Base@F@foo_inline#I#</USR><Declaration>virtual void foo_inline(int QQQ)</Declaration><Abstract><Para> Does something. </Para></Abstract><Parameters><Parameter><Name>QQQ</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> argument to defined virtual.</Para></Discussion></Parameter></Parameters></Function>]

// CHECK: FullCommentAsXML=[<Function file="{{[^"]+}}overriding-method-comments.mm" line="73" column="6"><Name>foo</Name><USR>c:@F@foo#I#</USR><Declaration>void foo(int DDD)</Declaration><Abstract><Para> Does something. </Para></Abstract><Parameters><Parameter><Name>DDD</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> a value.</Para></Discussion></Parameter></Parameters></Function>]

// CHECK: FullCommentAsXML=[<Function file="{{[^"]+}}overriding-method-comments.mm" line="75" column="6"><Name>foo</Name><USR>c:@F@foo#I#</USR><Declaration>void foo(int SSS)</Declaration><Abstract><Para> Does something. </Para></Abstract><Parameters><Parameter><Name>SSS</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> a value.</Para></Discussion></Parameter></Parameters></Function>]

// CHECK: FullCommentAsXML=[<Function file="{{[^"]+}}overriding-method-comments.mm" line="79" column="6"><Name>foo1</Name><USR>c:@F@foo1#I#</USR><Declaration>void foo1(int EEE)</Declaration><Abstract><Para> Does something. </Para></Abstract><Parameters><Parameter><Name>EEE</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> argument to function decl. </Para></Discussion></Parameter></Parameters></Function>]

// CHECK: FullCommentAsXML=[<Function file="{{[^"]+}}overriding-method-comments.mm" line="81" column="6"><Name>foo1</Name><USR>c:@F@foo1#I#</USR><Declaration>void foo1(int TTT)</Declaration><Abstract><Para> Does something. </Para></Abstract><Parameters><Parameter><Name>TTT</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> argument to function decl. </Para></Discussion></Parameter></Parameters></Function>]

// CHECK: FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}overriding-method-comments.mm" line="87" column="6"><Name>foo</Name><USR>c:@FT@&gt;2#T#Tfoo#t0.0#t0.1#</USR><Declaration>template &lt;typename AAA, typename BBB&gt; void foo(AAA, BBB)</Declaration><Abstract><Para> Documentation </Para></Abstract><TemplateParameters><Parameter><Name>AAA</Name><Index>0</Index><Discussion><Para> The type, silly as well.</Para></Discussion></Parameter><Parameter><Name>BBB</Name><Index>1</Index><Discussion><Para> The type, silly. </Para></Discussion></Parameter></TemplateParameters></Function>]

// CHECK: FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}overriding-method-comments.mm" line="90" column="6"><Name>foo</Name><USR>c:@FT@&gt;2#T#Tfoo#t0.0#t0.1#</USR><Declaration>template &lt;typename PPP, typename QQQ&gt; void foo(PPP, QQQ)</Declaration><Abstract><Para> Documentation </Para></Abstract><TemplateParameters><Parameter><Name>PPP</Name><Index>0</Index><Discussion><Para> The type, silly as well.</Para></Discussion></Parameter><Parameter><Name>QQQ</Name><Index>1</Index><Discussion><Para> The type, silly. </Para></Discussion></Parameter></TemplateParameters></Function>]
