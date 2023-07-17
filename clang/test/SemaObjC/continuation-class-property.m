// RUN: %clang_cc1  -fsyntax-only -verify %s
// radar 7509234

@protocol Foo
@property (readonly, copy) id foos;
@end

@interface Bar <Foo> {
}

@end

@interface Baz  <Foo> {
}
@end

@interface Bar ()
@property (readwrite, copy) id foos;
@end

@interface Baz ()
@property (readwrite, copy) id foos;
@end


// rdar://10142679
@class NSString;

typedef struct {
  float width;
  float length;
} NSRect;

@interface MyClass  {
}
@property (readonly) NSRect foo; // expected-note {{property declared here}}
@property (readonly, strong) NSString *bar; // expected-note {{property declared here}}
@end

@interface MyClass ()
@property (readwrite) NSString *foo; // expected-error {{type of property 'NSString *' in class extension does not match property type in primary class}}
@property (readwrite, strong) NSRect bar; // expected-error {{type of property 'NSRect' in class extension does not match property type in primary class}}
@end

// rdar://10655530
struct S;
struct S1;
@interface STAdKitContext
@property (nonatomic, readonly, assign) struct evhttp_request *httpRequest;
@property (nonatomic, readonly, assign) struct S *httpRequest2;
@property (nonatomic, readonly, assign) struct S1 *httpRequest3;
@property (nonatomic, readonly, assign) struct S2 *httpRequest4;
@end

struct evhttp_request;
struct S1;

@interface STAdKitContext()
@property (nonatomic, readwrite, assign) struct evhttp_request *httpRequest;
@property (nonatomic, readwrite, assign) struct S *httpRequest2;
@property (nonatomic, readwrite, assign) struct S1 *httpRequest3;
@property (nonatomic, readwrite, assign) struct S2 *httpRequest4;
@end

// rdar://15859862
@protocol ADCameraJSO_Bindings
@property (nonatomic, readonly) NSString *currentPictureURI;
@end

@interface ADCameraJSO
@end

@interface ADCameraJSO()  <ADCameraJSO_Bindings>
@property (nonatomic, copy) NSString *currentPictureURI;
@end
