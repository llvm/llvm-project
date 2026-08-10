// RUN: %clang_cc1 -verify %s

@interface Object 
@end

@protocol ProtocolObject
@property int class;
@property (copy) id MayCauseError;
@end

@protocol ProtocolDerivedGCObject <ProtocolObject>
@property int Dclass;
@end

@interface GCObject  : Object <ProtocolDerivedGCObject> {
    int ifield;
    int iOwnClass;
    int iDclass;
}
@property int OwnClass;
@end

@interface ReleaseObject : GCObject <ProtocolObject> {
   int newO;
   int oldO;
}
@property (retain) id MayCauseError;  // expected-warning {{'copy' attribute on property 'MayCauseError' does not match the property inherited from 'ProtocolObject'}}
@end

