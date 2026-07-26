#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <utility>
#include <unordered_map>

namespace {

uint64_t NextEventID = 0;
uint64_t CurrentEventID = 0;

// Теперь инструкция определяется парой ModuleToken, InstID -> последнее динамическое событие.
std::map<std::pair<uint64_t, uint64_t>, uint64_t>
    LastEventByInstruction;

// (Address, Size) -> EventID последнего store.
std::map<std::pair<uint64_t, uint64_t>, uint64_t>
    LastStoreEvent;

} // namespace

extern "C" void __def_use_trace_inst(uint64_t ModuleToken,
                                     uint64_t InstID) {
  CurrentEventID = NextEventID++;

  LastEventByInstruction[{ModuleToken, InstID}] = CurrentEventID;

  std::cerr << "EVENT "
            << CurrentEventID
            << " MODULE 0x"
            << std::hex
            << ModuleToken
            << std::dec
            << " INST "
            << InstID
            << '\n';
}

extern "C" void __def_use_trace_ssa_use(uint64_t ModuleToken,
                                        uint64_t DefInstID) {
  auto It =
      LastEventByInstruction.find({ModuleToken, DefInstID});

  if (It == LastEventByInstruction.end()) {
    return;
  }

  uint64_t DefEventID = It->second;

  std::cerr << "EDGE "
            << DefEventID
            << " -> "
            << CurrentEventID
            << '\n';
}

extern "C" void __def_use_trace_store(uint64_t Address,
                                      uint64_t Size) {
  std::cerr << "STORE 0x"
            << std::hex
            << Address
            << std::dec
            << " "
            << Size
            << '\n';

  std::pair<uint64_t, uint64_t> MemoryRange{Address, Size};

  LastStoreEvent[MemoryRange] = CurrentEventID;
}

extern "C" void __def_use_trace_load(uint64_t Address,
                                     uint64_t Size) {
  std::cerr << "LOAD 0x"
            << std::hex
            << Address
            << std::dec
            << " "
            << Size
            << '\n';

  std::pair<uint64_t, uint64_t> MemoryRange{Address, Size};

  auto It = LastStoreEvent.find(MemoryRange);

  if (It == LastStoreEvent.end()) {
    return;
  }

  uint64_t StoreEventID = It->second;

  std::cerr << "MEM_EDGE "
            << StoreEventID
            << " -> "
            << CurrentEventID
            << '\n';
}