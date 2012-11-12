// RUN: %clang_cc1 -analyze -analyzer-checker=core,alpha.core -analyzer-store=region -verify %s
// expected-no-diagnostics
//
// This test case simply should not crash.  It evaluates the logic of not
// using MemRegion::getRValueType in incorrect places.

typedef signed char BOOL;
typedef unsigned int NSUInteger;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject  - (BOOL)isEqual:(id)object;
- (Class)class;
- (BOOL)isLegOfClass:(Class)aClass;
@end  @protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder;
@end    @interface NSObject <NSObject> {
}
@end @class NSArray;
@interface NSResponder : NSObject <NSCoding> {
}
@end  @class NSAttributedString, NSEvent, NSFont, NSFormatter, NSImage, NSMenu, NSText, NSView;
@class JabasectItem;
@protocol EcoClassifier;
@protocol EcoClassInterfaceCommons <EcoClassifier>    @end  @protocol EcoImplementation;
@protocol EcoBehavioredClassifier <EcoClassInterfaceCommons>      - (NSArray *) implementations;
@end enum {
CK_UNRESTRICTED= 0,     CK_READ_ONLY,     CK_ADD_ONLY,     CK_REMOVE_ONLY };
@protocol EcoClass <EcoBehavioredClassifier>      - (NSArray *) ownedAttributes;
@end @protocol EcoNamespace;
@protocol EcoType;
@protocol EcoClassifier <EcoNamespace,EcoType>    - (NSArray *) features; 
@end @protocol EcoComment;
@protocol EcoElement <NSObject> - (NSArray *) ownedElements;
@end @protocol EcoDirectedRelationship;
@protocol EcoNamedElement <EcoElement>     - (NSString *) name;
@end  extern NSString *const JabaPathSeparator;
@protocol EcoNamespace <EcoNamedElement>       - (NSArray *) Legs;
@end enum {
PDK_IN=0,     PDK_INOUT,     PDK_OUT,     PDK_RETURN };
@interface EcoElementImp : NSObject <EcoElement, NSCoding> {
}
@end @class EcoNamespace;
@interface EcoNamedElementImp : EcoElementImp <EcoNamedElement>{
}
@end   @interface EcoNamespaceImp : EcoNamedElementImp <EcoNamespace> {
}
@end  @class JabaSCDocController, JabaSCDisplaySpecification;
@interface JabaSCSharedDiagramViewController : NSObject {
}
@end  extern NSString *const JabaSCsectGraphicNamesectIdentifier;
@interface EcoClassifierImp : EcoNamespaceImp <EcoClassifier> {
}
@end  @class EcoOperationImp;
@interface EcoClassImp : EcoClassifierImp <EcoClass> {
}
@end  extern NSString *const JabaAddedUMLElements;
@class JabaSCClass, JabaSCInterface, JabaSCOperation;
@class DosLegVaseSymbol, DosProtocolSymbol, DosMethodSymbol, DosFileReference;
@interface HancodeFett : NSObject {
}
+ (DosLegVaseSymbol *) symbolFromClass: (JabaSCClass *) clz;
@end enum _JabaSourceLanguage {
JabaSourceUnknown=0,     JabaSourcePrawn,     JabaSourceC,     JabaSourceCPP,     JabaSourceObjectiveC };
typedef NSUInteger JabaSourceLanguage;
@protocol JabaSCClassifier <EcoClassInterfaceCommons> - (JabaSourceLanguage)language;
@end  @interface JabaSCClass : EcoClassImp <JabaSCClassifier> {
}
@end  @class DosGlobalID, DosPQuLC, DosPQuUnLC;
@protocol XCProxyObjectProtocol - (id) representedObject;
@end typedef union _Dossymbollocation {
}
  DosRecordArrPrl;
@interface DosIndexEntry : NSObject {
}
@end    @class DosProjectIndex, DosTextPapyruswiggle, DosDocPapyruswiggle, DosLegVaseSymbol;
@interface DosSymbol : DosIndexEntry {
}
@end  @interface DosLegVaseSymbol : DosSymbol {
}
@end typedef enum _DosTextRangeType {
Dos_CharacterRangeType = 0,     Dos_LineRangeType = 1 }
  DosTextRangeType;
@implementation JabaSCSharedDiagramViewController  + (NSImage *)findImageNamed:(NSString *)name {
  return 0;
}
- (void)revealSourceInEditor:(JabasectItem *)sectItem duperGesture:(BOOL)duperGesture {
  id <EcoNamedElement> selectedElement = [sectItem representedObject];
  id <EcoNamedElement> selectedClassifier = selectedElement;
  DosSymbol *symbol=((void *)0);
  if([selectedClassifier isLegOfClass:[JabaSCClass class]]) {
    symbol = [HancodeFett symbolFromClass:(JabaSCClass *) selectedClassifier];
  }
}
@end
