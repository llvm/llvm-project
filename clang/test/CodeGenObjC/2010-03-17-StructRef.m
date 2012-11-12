// RUN: %clang_cc1 %s -emit-llvm -triple x86_64-apple-darwin -fobjc-runtime=macosx-fragile-10.5 -o - | FileCheck %s
// Bitfield references must not touch memory outside of the enclosing
// struct.   Radar 7639995
typedef signed char BOOL;
@protocol NSObject
- (id)init;
@end
@interface NSObject <NSObject> {}
@end
@interface IMAVChatParticipant : NSObject {
  int _ardRole;
  int _state;
  int _avRelayStatus;
  int _chatEndedReason;
  int _chatError;
  unsigned _sendingAudio:1;
  unsigned _sendingVideo:1;
  unsigned _sendingAuxVideo:1;
  unsigned _audioMuted:1;
  unsigned _videoPaused:1;
  unsigned _networkStalled:1;
  unsigned _isInitiator:1;
  unsigned _isAOLInterop:1;
  unsigned _isRecording:1;
  unsigned _isUsingICE:1;
}
@end
@implementation IMAVChatParticipant
- (id) init {
  self = [super init];
  if ( self ) {
    BOOL blah = (BOOL)1;
    // We're expecting these three bitfield assignments will generate i8 stores.
    _sendingAudio = (BOOL)1;
    _isUsingICE = (BOOL)1;
    _isUsingICE = blah;
    // CHECK: store i8
    // CHECK: store i8
    // CHECK: store i8
  }
  return self;
}
@end
