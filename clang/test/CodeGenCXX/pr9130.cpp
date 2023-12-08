// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

class nsOggCodecState {
  virtual int StartTime() {
    return -1;
  }
};
class nsVorbisState : public nsOggCodecState {
  virtual ~nsVorbisState();
};
nsVorbisState::~nsVorbisState() {
}

// CHECK-LABEL: define linkonce_odr noundef i32 @_ZN15nsOggCodecState9StartTimeEv
