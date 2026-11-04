// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng -target x86_64-apple-macosx26.0 %s > %t/out
// RUN: FileCheck %s < %t/out

// Ensure that XML we generate is not invalid.
// RUN: FileCheck %s -check-prefix=WRONG < %t/out
// WRONG-NOT: CommentXMLInvalid

// Test that anyAppleOS availability emits two <Availability> entries in XML:
// one for the original anyAppleOS attr and one for the inferred platform attr.

/// Aaa.
void attr_anyappleos_1() __attribute__((availability(anyAppleOS, introduced=26.0)));

// CHECK: FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments-availability-anyappleos.cpp" line="[[@LINE-2]]" column="6"><Name>attr_anyappleos_1</Name><USR>c:@F@attr_anyappleos_1#</USR><Declaration>void attr_anyappleos_1()</Declaration><Abstract><Para> Aaa.</Para></Abstract><Availability distribution="any Apple OS"><IntroducedInVersion>26.0</IntroducedInVersion></Availability><Availability distribution="macOS"><IntroducedInVersion>26.0</IntroducedInVersion></Availability></Function>]

/// Aaa.
void attr_anyappleos_2() __attribute__((availability(anyAppleOS, introduced=26.0, deprecated=27.0, message="use something else")));

// CHECK: FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments-availability-anyappleos.cpp" line="[[@LINE-2]]" column="6"><Name>attr_anyappleos_2</Name><USR>c:@F@attr_anyappleos_2#</USR><Declaration>void attr_anyappleos_2()</Declaration><Abstract><Para> Aaa.</Para></Abstract><Availability distribution="any Apple OS"><IntroducedInVersion>26.0</IntroducedInVersion><DeprecatedInVersion>27.0</DeprecatedInVersion><DeprecationSummary>use something else</DeprecationSummary></Availability><Availability distribution="macOS"><IntroducedInVersion>26.0</IntroducedInVersion><DeprecatedInVersion>27.0</DeprecatedInVersion><DeprecationSummary>use something else</DeprecationSummary></Availability></Function>]

/// Aaa.
void attr_anyappleos_3() __attribute__((availability(anyAppleOS, unavailable, message="not available")));

// CHECK: FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments-availability-anyappleos.cpp" line="[[@LINE-2]]" column="6"><Name>attr_anyappleos_3</Name><USR>c:@F@attr_anyappleos_3#</USR><Declaration>void attr_anyappleos_3()</Declaration><Abstract><Para> Aaa.</Para></Abstract><Availability distribution="any Apple OS"><DeprecationSummary>not available</DeprecationSummary><Unavailable/></Availability><Availability distribution="macOS"><DeprecationSummary>not available</DeprecationSummary><Unavailable/></Availability></Function>]
