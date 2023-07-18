// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng %s > %t/out
// RUN: FileCheck %s < %t/out
// rdar://13067629

// Ensure that XML we generate is not invalid.
// RUN: FileCheck %s -check-prefix=WRONG < %t/out
// WRONG-NOT: CommentXMLInvalid

// rdar://12397511

/*!
     \headerfile Device.h <Foundation/Device.h>

      A Device represents a remote or local computer or device with which the Developer Tools can interact.  Each Device supports blah blah blah from doing blah blah blah.
*/
@interface Device
@end
// CHECK: headerfile-comment-to-html.m:[[@LINE-2]]:12: ObjCInterfaceDecl=Device:{{.*}} FullCommentAsXML=[<Other file="{{[^"]+}}headerfile-comment-to-html.m" line="[[@LINE-2]]" column="12"><Name>Device</Name><USR>c:objc(cs)Device</USR><Headerfile><Para> Device.h &lt;Foundation/Device.h&gt;</Para></Headerfile><Declaration>@interface Device\n@end</Declaration><Abstract><Para>      A Device represents a remote or local computer or device with which the Developer Tools can interact.  Each Device supports blah blah blah from doing blah blah blah.</Para></Abstract></Other>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[     ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[headerfile]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Device.h ])
// CHECK-NEXT:           (CXComment_Text Text=[<Foundation])
// CHECK-NEXT:           (CXComment_Text Text=[/Device.h>])))
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[      A Device represents a remote or local computer or device with which the Developer Tools can interact.  Each Device supports blah blah blah from doing blah blah blah.])))]

/*!
    \headerfile Sensor.h "Sensor.h"

    \brief This is Sensor on the Device.
    Its purpose is not to Sense Device's heat.
*/

@interface Sensor
@end
// CHECK: headerfile-comment-to-html.m:[[@LINE-2]]:12: ObjCInterfaceDecl=Sensor:{{.*}} FullCommentAsXML=[<Other file="{{[^"]+}}headerfile-comment-to-html.m" line="[[@LINE-2]]" column="12"><Name>Sensor</Name><USR>c:objc(cs)Sensor</USR><Headerfile><Para> Sensor.h &quot;Sensor.h&quot;</Para></Headerfile><Declaration>@interface Sensor\n@end</Declaration><Abstract><Para> This is Sensor on the Device.    Its purpose is not to Sense Device&apos;s heat.</Para></Abstract></Other>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[    ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[headerfile]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Sensor.h "Sensor.h"])))
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[    ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[brief]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ This is Sensor on the Device.] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[    Its purpose is not to Sense Device's heat.]))))]

/*!
    \brief Test that headerfile can come after brief.
    \headerfile VTDevice.h <VTFoundation/VTDevice.h>

    More property description goes here.
*/
@interface VTDevice : Device
@end
// CHECK: headerfile-comment-to-html.m:[[@LINE-2]]:12: ObjCInterfaceDecl=VTDevice:{{.*}} FullCommentAsXML=[<Other file="{{[^"]+}}headerfile-comment-to-html.m" line="[[@LINE-2]]" column="12"><Name>VTDevice</Name><USR>c:objc(cs)VTDevice</USR><Headerfile><Para> VTDevice.h &lt;VTFoundation/VTDevice.h&gt;</Para></Headerfile><Declaration>@interface VTDevice : Device\n@end</Declaration><Abstract><Para> Test that headerfile can come after brief.    </Para></Abstract><Discussion><Para>    More property description goes here.</Para></Discussion></Other>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[    ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[brief]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Test that headerfile can come after brief.] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[    ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[headerfile]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ VTDevice.h ])
// CHECK-NEXT:           (CXComment_Text Text=[<VTFoundation])
// CHECK-NEXT:           (CXComment_Text Text=[/VTDevice.h>])))
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[    More property description goes here.])))]

/*!
  \headerfile  <stdio.h>
*/
extern void uses_stdio_h();
// CHECK: headerfile-comment-to-html.m:[[@LINE-1]]:13: FunctionDecl=uses_stdio_h:{{.*}} FullCommentAsXML=[<Function file="{{[^"]+}}headerfile-comment-to-html.m" line="[[@LINE-1]]" column="13"><Name>uses_stdio_h</Name><USR>c:@F@uses_stdio_h</USR><Headerfile><Para>  &lt;stdio.h&gt;</Para></Headerfile><Declaration>extern void uses_stdio_h()</Declaration></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[  ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[headerfile]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[  ] IsWhitespace)
// CHECK-NEXT:           (CXComment_Text Text=[<stdio])
// CHECK-NEXT:           (CXComment_Text Text=[.h>]))))]


/*!
  \headerfile  <algorithm>
*/
extern void uses_argorithm();
// CHECK: headerfile-comment-to-html.m:[[@LINE-1]]:13: FunctionDecl=uses_argorithm:{{.*}} FullCommentAsXML=[<Function file="{{[^"]+}}headerfile-comment-to-html.m" line="[[@LINE-1]]" column="13"><Name>uses_argorithm</Name><USR>c:@F@uses_argorithm</USR><Headerfile><Para>  &lt;algorithm&gt;</Para></Headerfile><Declaration>extern void uses_argorithm()</Declaration></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[  ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[headerfile]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[  ] IsWhitespace)
// CHECK-NEXT:           (CXComment_Text Text=[<algorithm])
// CHECK-NEXT:           (CXComment_Text Text=[>]))))]
