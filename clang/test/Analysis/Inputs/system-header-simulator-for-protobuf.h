// Like the compiler, the static analyzer treats some functions differently if
// they come from a system header -- for example, it is assumed that system
// functions do not arbitrarily free() their parameters, and that some bugs
// found in system headers cannot be fixed by the user and should be
// suppressed.
#pragma clang system_header

class Arena;
class MessageLite {
  int SomeArbitraryField;
};

// Originally declared in generated_message_util.h
MessageLite *GetOwnedMessageInternal(Arena *, MessageLite *, Arena *);

// Not a real protobuf function -- just introduced to validate that this file
// is handled as a system header.
void SomeOtherFunction(MessageLite *);
