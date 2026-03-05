
#ifndef IPC_MANAGER_BS_HPP
#define IPC_MANAGER_BS_HPP

#include "clang/IPC2978/Manager.hpp"
#include "clang/IPC2978/Messages.hpp"

namespace P2978
{

// IPC Manager BuildSystem
class IPCManagerBS : public Manager
{
  public:
    uint64_t writeFd = 0;

    tl::expected<void, std::string> writeInternal(std::string_view buffer) const override;

    explicit IPCManagerBS(uint64_t writeFd_);
    static tl::expected<void, std::string> receiveMessage(char (&ctbBuffer)[320], CTB &messageType, std::string_view serverReadString) ;
    [[nodiscard]] tl::expected<void, std::string> sendMessage(const BTCModule &moduleFile) const;
    [[nodiscard]] tl::expected<void, std::string> sendMessage(const BTCNonModule &nonModule) const;
    [[nodiscard]] tl::expected<void, std::string> sendMessage(const BTCLastMessage &lastMessage) const;
    static tl::expected<Mapping, std::string> createSharedMemoryBMIFile(BMIFile &bmiFile);
    static tl::expected<void, std::string> closeBMIFileMapping(const Mapping &processMappingOfBMIFile);
};
} // namespace P2978
#endif // IPC_MANAGER_BS_HPP
