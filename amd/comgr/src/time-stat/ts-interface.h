#ifndef AMD_COMGR_TS_INTERFACE_H
#define AMD_COMGR_TS_INTERFACE_H

// External interface

namespace TimeStatistics {

bool InitTimeStatistics(std::string LogFile);
void StartAction(amd_comgr_action_kind_t);
void EndAction();

} // namespace TimeStatistics

#endif // AMD_COMGR_TS_INTERFACE_H
