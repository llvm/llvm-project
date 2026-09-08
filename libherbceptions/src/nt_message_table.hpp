// clang-format off
// Generated from utils/ntkernel-table.json; do not edit.
// NTSTATUS -> US-English UTF-8 message scatter. Requires <herbceptions/error>.
	case 0x0u:
		return __tsc(u8"The operation completed successfully.");
	case 0x1u:
	case 0x2u:
	case 0x3u:
	case 0x3fu:
		return __tsc(u8"The caller specified WaitAny for WaitType and one of the dispatcher objects in the Object array has been set to the signaled state.");
	case 0x80u:
	case 0xbfu:
		return __tsc(u8"The caller attempted to wait for a mutex that has been abandoned.");
	case 0xc0u:
		return __tsc(u8"A user-mode APC was delivered before the given Interval expired.");
	case 0xffu:
	case 0x100u:
		return __tsc(u8"");
	case 0x101u:
		return __tsc(u8"The delay completed because the thread was alerted.");
	case 0x102u:
		return __tsc(u8"The given Timeout interval expired.");
	case 0x103u:
		return __tsc(u8"The operation that was requested is pending completion.");
	case 0x104u:
		return __tsc(u8"A reparse should be performed by the Object Manager because the name of the file resulted in a symbolic link.");
	case 0x105u:
		return __tsc(u8"Returned by enumeration APIs to indicate more information is available to successive calls.");
	case 0x106u:
		return __tsc(u8"Indicates not all privileges or groups that are referenced are assigned to the caller. This allows, for example, all privileges to be disabled without having to know exactly which privileges are assigned.");
	case 0x107u:
		return __tsc(u8"Some of the information to be translated has not been translated.");
	case 0x108u:
		return __tsc(u8"An open/create operation completed while an opportunistic lock (oplock) break is underway.");
	case 0x109u:
		return __tsc(u8"A new volume has been mounted by a file system.");
	case 0x10au:
		return __tsc(u8"This success level status indicates that the transaction state already exists for the registry subtree but that a transaction commit was previously aborted. The commit has now been completed.");
	case 0x10bu:
		return __tsc(u8"Indicates that a notify change request has been completed due to closing the handle that made the notify change request.");
	case 0x10cu:
		return __tsc(u8"Indicates that a notify change request is being completed and that the information is not being returned in the caller's buffer. The caller now needs to enumerate the files to find the changes.");
	case 0x10du:
		return __tsc(u8"{No Quotas} No system quota limits are specifically set for this account.");
	case 0x10eu:
		return __tsc(u8"{Connect Failure on Primary Transport} An attempt was made to connect to the remote server %hs on the primary transport, but the connection failed. The computer WAS able to connect on a secondary transport.");
	case 0x110u:
		return __tsc(u8"The page fault was a transition fault.");
	case 0x111u:
	case 0x112u:
	case 0x113u:
		return __tsc(u8"The page fault was a demand zero fault.");
	case 0x114u:
		return __tsc(u8"The page fault was satisfied by reading from a secondary storage device.");
	case 0x115u:
		return __tsc(u8"The cached page was locked during operation.");
	case 0x116u:
		return __tsc(u8"The crash dump exists in a paging file.");
	case 0x117u:
		return __tsc(u8"The specified buffer contains all zeros.");
	case 0x118u:
		return __tsc(u8"A reparse should be performed by the Object Manager because the name of the file resulted in a symbolic link.");
	case 0x119u:
		return __tsc(u8"The device has succeeded a query-stop and its resource requirements have changed.");
	case 0x120u:
		return __tsc(u8"The translator has translated these resources into the global space and no additional translations should be performed.");
	case 0x121u:
		return __tsc(u8"The directory service evaluated group memberships locally, because it was unable to contact a global catalog server.");
	case 0x122u:
		return __tsc(u8"A process being terminated has no threads to terminate.");
	case 0x123u:
		return __tsc(u8"The specified process is not part of a job.");
	case 0x124u:
		return __tsc(u8"The specified process is part of a job.");
	case 0x125u:
		return __tsc(u8"{Volume Shadow Copy Service} The system is now ready for hibernation.");
	case 0x126u:
		return __tsc(u8"A file system or file system filter driver has successfully completed an FsFilter operation.");
	case 0x127u:
		return __tsc(u8"The specified interrupt vector was already connected.");
	case 0x128u:
		return __tsc(u8"The specified interrupt vector is still connected.");
	case 0x129u:
		return __tsc(u8"The current process is a cloned process.");
	case 0x12au:
		return __tsc(u8"The file was locked and all users of the file can only read.");
	case 0x12bu:
		return __tsc(u8"The file was locked and at least one user of the file can write.");
	case 0x12cu:
	case 0x12du:
	case 0x12eu:
	case 0x12fu:
	case 0x130u:
		return __tsc(u8"");
	case 0x202u:
		return __tsc(u8"The specified ResourceManager made no changes or updates to the resource under this transaction.");
	case 0x210u:
	case 0x211u:
	case 0x212u:
	case 0x213u:
	case 0x214u:
	case 0x215u:
	case 0x216u:
		return __tsc(u8"");
	case 0x367u:
		return __tsc(u8"An operation is blocked and waiting for an oplock.");
	case 0x368u:
	case 0x369u:
		return __tsc(u8"");
	case 0x10001u:
		return __tsc(u8"Debugger handled the exception.");
	case 0x10002u:
		return __tsc(u8"The debugger continued.");
	case 0x1c0001u:
		return __tsc(u8"The IO was completed by a filter.");
	case 0x293000u:
	case 0x293001u:
	case 0x350059u:
	case 0xe70000u:
	case 0xe70001u:
	case 0xe70002u:
	case 0xe70003u:
		return __tsc(u8"");
	case 0x40000000u:
		return __tsc(u8"{Object Exists} An attempt was made to create an object but the object name already exists.");
	case 0x40000001u:
		return __tsc(u8"{Thread Suspended} A thread termination occurred while the thread was suspended. The thread resumed, and termination proceeded.");
	case 0x40000002u:
		return __tsc(u8"{Working Set Range Error} An attempt was made to set the working set minimum or maximum to values that are outside the allowable range.");
	case 0x40000003u:
		return __tsc(u8"{Image Relocated} An image file could not be mapped at the address that is specified in the image file. Local fixes must be performed on this image.");
	case 0x40000004u:
		return __tsc(u8"This informational level status indicates that a specified registry subtree transaction state did not yet exist and had to be created.");
	case 0x40000005u:
		return __tsc(u8"{Segment Load} A virtual DOS machine (VDM) is loading, unloading, or moving an MS-DOS or Win16 program segment image. An exception is raised so that a debugger can load, unload, or track symbols and breakpoints within these 16-bit segments.");
	case 0x40000006u:
		return __tsc(u8"{Local Session Key} A user session key was requested for a local remote procedure call (RPC) connection. The session key that is returned is a constant value and not unique to this connection.");
	case 0x40000007u:
		return __tsc(u8"{Invalid Current Directory} The process cannot switch to the startup current directory %hs. Select OK to set the current directory to %hs, or select CANCEL to exit.");
	case 0x40000008u:
		return __tsc(u8"{Serial IOCTL Complete} A serial I/O operation was completed by another write to a serial port. (The IOCTL\\_SERIAL\\_XOFF\\_COUNTER reached zero.)");
	case 0x40000009u:
		return __tsc(u8"{Registry Recovery} One of the files that contains the system registry data had to be recovered by using a log or alternate copy. The recovery was successful.");
	case 0x4000000au:
		return __tsc(u8"{Redundant Read} To satisfy a read request, the Windows NT operating system fault-tolerant file system successfully read the requested data from a redundant copy. This was done because the file system encountered a failure on a member of the fault-tolerant volume but was unable to reassign the failing area of the device.");
	case 0x4000000bu:
		return __tsc(u8"{Redundant Write} To satisfy a write request, the Windows NT fault-tolerant file system successfully wrote a redundant copy of the information. This was done because the file system encountered a failure on a member of the fault-tolerant volume but was unable to reassign the failing area of the device.");
	case 0x4000000cu:
		return __tsc(u8"{Serial IOCTL Timeout} A serial I/O operation completed because the time-out period expired. (The IOCTL\\_SERIAL\\_XOFF\\_COUNTER had not reached zero.)");
	case 0x4000000du:
		return __tsc(u8"{Password Too Complex} The Windows password is too complex to be converted to a LAN Manager password. The LAN Manager password that returned is a NULL string.");
	case 0x4000000eu:
		return __tsc(u8"{Machine Type Mismatch} The image file %hs is valid but is for a machine type other than the current machine. Select OK to continue, or CANCEL to fail the DLL load.");
	case 0x4000000fu:
		return __tsc(u8"{Partial Data Received} The network transport returned partial data to its client. The remaining data will be sent later.");
	case 0x40000010u:
		return __tsc(u8"{Expedited Data Received} The network transport returned data to its client that was marked as expedited by the remote system.");
	case 0x40000011u:
		return __tsc(u8"{Partial Expedited Data Received} The network transport returned partial data to its client and this data was marked as expedited by the remote system. The remaining data will be sent later.");
	case 0x40000012u:
		return __tsc(u8"{TDI Event Done} The TDI indication has completed successfully.");
	case 0x40000013u:
		return __tsc(u8"{TDI Event Pending} The TDI indication has entered the pending state.");
	case 0x40000014u:
		return __tsc(u8"Checking file system on %wZ.");
	case 0x40000015u:
		return __tsc(u8"{Fatal Application Exit} %hs");
	case 0x40000016u:
		return __tsc(u8"The specified registry key is referenced by a predefined handle.");
	case 0x40000017u:
		return __tsc(u8"{Page Unlocked} The page protection of a locked page was changed to 'No Access' and the page was unlocked from memory and from the process.");
	case 0x40000018u:
		return __tsc(u8"%hs");
	case 0x40000019u:
		return __tsc(u8"{Page Locked} One of the pages to lock was already locked.");
	case 0x4000001au:
		return __tsc(u8"Application popup: %1 : %2");
	case 0x4000001bu:
		return __tsc(u8"A Win32 process already exists.");
	case 0x4000001cu:
	case 0x4000001du:
	case 0x4000001eu:
	case 0x4000001fu:
	case 0x40000020u:
	case 0x40000021u:
	case 0x40000022u:
		return __tsc(u8"An exception status code that is used by the Win32 x86 emulation subsystem.");
	case 0x40000023u:
		return __tsc(u8"{Machine Type Mismatch} The image file %hs is valid but is for a machine type other than the current machine.");
	case 0x40000024u:
		return __tsc(u8"A yield execution was performed and no thread was available to run.");
	case 0x40000025u:
		return __tsc(u8"The resume flag to a timer API was ignored.");
	case 0x40000026u:
		return __tsc(u8"The arbiter has deferred arbitration of these resources to its parent.");
	case 0x40000027u:
		return __tsc(u8"The device has detected a CardBus card in its slot.");
	case 0x40000028u:
		return __tsc(u8"An exception status code that is used by the Win32 x86 emulation subsystem.");
	case 0x40000029u:
		return __tsc(u8"The CPUs in this multiprocessor system are not all the same revision level. To use all processors, the operating system restricts itself to the features of the least capable processor in the system. If problems occur with this system, contact the CPU manufacturer to see if this mix of processors is supported.");
	case 0x4000002au:
		return __tsc(u8"The system was put into hibernation.");
	case 0x4000002bu:
		return __tsc(u8"The system was resumed from hibernation.");
	case 0x4000002cu:
		return __tsc(u8"Windows has detected that the system firmware (BIOS) was updated [previous firmware date = %2, current firmware date %3].");
	case 0x4000002du:
		return __tsc(u8"A device driver is leaking locked I/O pages and is causing system degradation. The system has automatically enabled the tracking code to try and catch the culprit.");
	case 0x4000002eu:
		return __tsc(u8"The ALPC message being canceled has already been retrieved from the queue on the other side.");
	case 0x4000002fu:
		return __tsc(u8"The system power state is transitioning from %2 to %3.");
	case 0x40000030u:
		return __tsc(u8"The receive operation was successful. Check the ALPC completion list for the received message.");
	case 0x40000031u:
		return __tsc(u8"The system power state is transitioning from %2 to %3 but could enter %4.");
	case 0x40000032u:
		return __tsc(u8"Access to %1 is monitored by policy rule %2.");
	case 0x40000033u:
		return __tsc(u8"A valid hibernation file has been invalidated and should be abandoned.");
	case 0x40000034u:
		return __tsc(u8"Business rule scripts are disabled for the calling application.");
	case 0x40000035u:
	case 0x40000036u:
	case 0x40000037u:
	case 0x40000038u:
	case 0x40000039u:
		return __tsc(u8"");
	case 0x40000294u:
		return __tsc(u8"The system has awoken.");
	case 0x40000370u:
		return __tsc(u8"The directory service is shutting down.");
	case 0x40000807u:
	case 0x4000a144u:
		return __tsc(u8"");
	case 0x40010001u:
		return __tsc(u8"Debugger will reply later.");
	case 0x40010002u:
		return __tsc(u8"Debugger cannot provide a handle.");
	case 0x40010003u:
		return __tsc(u8"Debugger terminated the thread.");
	case 0x40010004u:
		return __tsc(u8"Debugger terminated the process.");
	case 0x40010005u:
		return __tsc(u8"Debugger obtained control of C.");
	case 0x40010006u:
		return __tsc(u8"Debugger printed an exception on control C.");
	case 0x40010007u:
		return __tsc(u8"Debugger received a RIP exception.");
	case 0x40010008u:
		return __tsc(u8"Debugger received a control break.");
	case 0x40010009u:
		return __tsc(u8"Debugger command communication exception.");
	case 0x4001000au:
		return __tsc(u8"");
	case 0x40020056u:
		return __tsc(u8"A UUID that is valid only on this computer has been allocated.");
	case 0x400200afu:
		return __tsc(u8"Some data remains to be sent in the request buffer.");
	case 0x400a0004u:
		return __tsc(u8"The Client Drive Mapping Service has connected on Terminal Connection.");
	case 0x400a0005u:
		return __tsc(u8"The Client Drive Mapping Service has disconnected on Terminal Connection.");
	case 0x4015000du:
		return __tsc(u8"A kernel mode component is releasing a reference on an activation context.");
	case 0x40190001u:
		return __tsc(u8"");
	case 0x40190034u:
		return __tsc(u8"The transactional resource manager is already consistent. Recovery is not needed.");
	case 0x40190035u:
		return __tsc(u8"The transactional resource manager has already been started.");
	case 0x401a000cu:
		return __tsc(u8"The log service encountered a log stream with no restart area.");
	case 0x401b00ecu:
		return __tsc(u8"{Display Driver Recovered From Failure} The %hs display driver has detected a failure and recovered from it. Some graphical operations might have failed. The next time you restart the machine, a dialog box appears, giving you an opportunity to upload data about this failure to Microsoft.");
	case 0x401e000au:
		return __tsc(u8"The specified buffer is not big enough to contain the entire requested dataset. Partial data is populated up to the size of the buffer.\nThe caller needs to provide a buffer of the size as specified in the partially populated buffer's content (interface specific).");
	case 0x401e0201u:
		return __tsc(u8"");
	case 0x401e0307u:
		return __tsc(u8"No mode is pinned on the specified VidPN source/target.");
	case 0x401e031eu:
		return __tsc(u8"The specified mode set does not specify a preference for one of its modes.");
	case 0x401e034bu:
		return __tsc(u8"The specified dataset (for example, mode set, frequency range set, descriptor set, or topology) is empty.");
	case 0x401e034cu:
		return __tsc(u8"The specified dataset (for example, mode set, frequency range set, descriptor set, or topology) does not contain any more elements.");
	case 0x401e0351u:
		return __tsc(u8"The specified content transformation is not pinned on the specified VidPN present path.");
	case 0x401e042fu:
		return __tsc(u8"The child device presence was not reliably detected.");
	case 0x401e0437u:
		return __tsc(u8"Starting the lead adapter in a linked configuration has been temporarily deferred.");
	case 0x401e0439u:
		return __tsc(u8"The display adapter is being polled for children too frequently at the same polling level.");
	case 0x401e043au:
		return __tsc(u8"Starting the adapter has been temporarily deferred.");
	case 0x401e043cu:
		return __tsc(u8"");
	case 0x40230001u:
		return __tsc(u8"The request will be completed later by an NDIS status indication.");
	case 0x40292023u:
		return __tsc(u8"");
	case 0x80000001u:
		return __tsc(u8"{EXCEPTION}\nGuard Page Exception\nA page of memory that marks the end of a data structure, such as a stack or an array, has been accessed.");
	case 0x80000002u:
		return __tsc(u8"{EXCEPTION}\nAlignment Fault\nA datatype misalignment was detected in a load or store instruction.");
	case 0x80000003u:
		return __tsc(u8"{EXCEPTION}\nBreakpoint\nA breakpoint has been reached.");
	case 0x80000004u:
		return __tsc(u8"{EXCEPTION}\nSingle Step\nA single step or trace operation has just been completed.");
	case 0x80000005u:
		return __tsc(u8"{Buffer Overflow}\nThe data was too large to fit into the specified buffer.");
	case 0x80000006u:
		return __tsc(u8"{No More Files}\nNo more files were found which match the file specification.");
	case 0x80000007u:
		return __tsc(u8"{Kernel Debugger Awakened}\nthe system debugger was awakened by an interrupt.");
	case 0x8000000au:
		return __tsc(u8"{Handles Closed}\nHandles to objects have been automatically closed as a result of the requested operation.");
	case 0x8000000bu:
		return __tsc(u8"{Non-Inheritable ACL}\nAn access control list (ACL) contains no components that can be inherited.");
	case 0x8000000cu:
		return __tsc(u8"{GUID Substitution}\nDuring the translation of a global identifier (GUID) to a Windows security ID (SID), no administratively-defined GUID prefix was found. A substitute prefix was used, which will not compromise system security. However, this may provide a more restrictive access than intended.");
	case 0x8000000du:
		return __tsc(u8"{Partial Copy}\nDue to protection conflicts not all the requested bytes could be copied.");
	case 0x8000000eu:
		return __tsc(u8"{Out of Paper}\nThe printer is out of paper.");
	case 0x8000000fu:
		return __tsc(u8"{Device Power Is Off}\nThe printer power has been turned off.");
	case 0x80000010u:
		return __tsc(u8"{Device Offline}\nThe printer has been taken offline.");
	case 0x80000011u:
		return __tsc(u8"{Device Busy}\nThe device is currently busy.");
	case 0x80000012u:
		return __tsc(u8"{No More EAs}\nNo more extended attributes (EAs) were found for the file.");
	case 0x80000013u:
		return __tsc(u8"{Illegal EA}\nThe specified extended attribute (EA) name contains at least one illegal character.");
	case 0x80000014u:
		return __tsc(u8"{Inconsistent EA List}\nThe extended attribute (EA) list is inconsistent.");
	case 0x80000015u:
		return __tsc(u8"{Invalid EA Flag}\nAn invalid extended attribute (EA) flag was set.");
	case 0x80000016u:
		return __tsc(u8"{Verifying Disk}\nThe media has changed and a verify operation is in progress so no reads or writes may be performed to the device, except those used in the verify operation.");
	case 0x80000017u:
		return __tsc(u8"{Too Much Information}\nThe specified access control list (ACL) contained more information than was expected.");
	case 0x80000018u:
		return __tsc(u8"This warning level status indicates that the transaction state already exists for the registry sub-tree, but that a transaction commit was previously aborted.\nThe commit has NOT been completed, but has not been rolled back either (so it may still be committed if desired).");
	case 0x8000001au:
		return __tsc(u8"{No More Entries}\nNo more entries are available from an enumeration operation.");
	case 0x8000001bu:
		return __tsc(u8"{Filemark Found}\nA filemark was detected.");
	case 0x8000001cu:
		return __tsc(u8"{Media Changed}\nThe media may have changed.");
	case 0x8000001du:
		return __tsc(u8"{I/O Bus Reset}\nAn I/O bus reset was detected.");
	case 0x8000001eu:
		return __tsc(u8"{End of Media}\nThe end of the media was encountered.");
	case 0x8000001fu:
		return __tsc(u8"Beginning of tape or partition has been detected.");
	case 0x80000020u:
		return __tsc(u8"{Media Changed}\nThe media may have changed.");
	case 0x80000021u:
		return __tsc(u8"A tape access reached a setmark.");
	case 0x80000022u:
		return __tsc(u8"During a tape access, the end of the data written is reached.");
	case 0x80000023u:
		return __tsc(u8"The redirector is in use and cannot be unloaded.");
	case 0x80000024u:
		return __tsc(u8"The server is in use and cannot be unloaded.");
	case 0x80000025u:
		return __tsc(u8"The specified connection has already been disconnected.");
	case 0x80000026u:
		return __tsc(u8"A long jump has been executed.");
	case 0x80000027u:
		return __tsc(u8"A cleaner cartridge is present in the tape library.");
	case 0x80000028u:
		return __tsc(u8"The Plug and Play query operation was not successful.");
	case 0x80000029u:
		return __tsc(u8"A frame consolidation has been executed.");
	case 0x8000002au:
		return __tsc(u8"{Registry Hive Recovered}\nRegistry hive (file):\n%hs\nwas corrupted and it has been recovered. Some data might have been lost.");
	case 0x8000002bu:
		return __tsc(u8"The application is attempting to run executable code from the module %hs. This may be insecure. An alternative, %hs, is available. Should the application use the secure module %hs?");
	case 0x8000002cu:
		return __tsc(u8"The application is loading executable code from the module %hs. This is secure, but may be incompatible with previous releases of the operating system. An alternative, %hs, is available. Should the application use the secure module %hs?");
	case 0x8000002du:
		return __tsc(u8"The create operation stopped after reaching a symbolic link.");
	case 0x8000002eu:
		return __tsc(u8"An oplock of the requested level cannot be granted.  An oplock of a lower level may be available.");
	case 0x8000002fu:
		return __tsc(u8"{No ACE Condition}\nThe specified access control entry (ACE) does not contain a condition.");
	case 0x80000030u:
		return __tsc(u8"{Support Being Determined}\nDevice's command support detection is in progress.");
	case 0x80000031u:
		return __tsc(u8"The device needs to be power cycled.");
	case 0x80000032u:
		return __tsc(u8"The action requested resulted in no work being done. Error-style clean-up has been performed.");
	case 0x80000033u:
	case 0x80000034u:
	case 0x800001b6u:
		return __tsc(u8"");
	case 0x80000288u:
		return __tsc(u8"The device has indicated that cleaning is necessary.");
	case 0x80000289u:
		return __tsc(u8"The device has indicated that its door is open. Further operations require it closed and secured.");
	case 0x80000803u:
		return __tsc(u8"Windows discovered a corruption in the file \"%hs\".\nThis file has now been repaired.\nPlease check if any data in the file was lost because of the corruption.");
	case 0x8000a127u:
		return __tsc(u8"The interrupt requested to be unmasked is not masked.");
	case 0x8000cf00u:
		return __tsc(u8"The Cloud File property blob is possibly corrupt. The on-disk checksum does not match the computed checksum.");
	case 0x8000cf04u:
		return __tsc(u8"The operation could not be completed because the Cloud File property blob is too large.");
	case 0x8000cf05u:
		return __tsc(u8"The operation could not be completed because the maximum number of Cloud File property blobs would be exceeded.");
	case 0x80010001u:
		return __tsc(u8"Debugger did not handle the exception.");
	case 0x80130001u:
		return __tsc(u8"The cluster node is already up.");
	case 0x80130002u:
		return __tsc(u8"The cluster node is already down.");
	case 0x80130003u:
		return __tsc(u8"The cluster network is already online.");
	case 0x80130004u:
		return __tsc(u8"The cluster network is already offline.");
	case 0x80130005u:
		return __tsc(u8"The cluster node is already a member of the cluster.");
	case 0x80190009u:
		return __tsc(u8"The log could not be set to the requested size.");
	case 0x80190029u:
		return __tsc(u8"There is no transaction metadata on the file.");
	case 0x80190031u:
		return __tsc(u8"The file cannot be recovered because there is a handle still open on it.");
	case 0x80190041u:
		return __tsc(u8"Transaction metadata is already present on this file and cannot be superseded.");
	case 0x80190042u:
		return __tsc(u8"A transaction scope could not be entered because the scope handler has not been initialized.");
	case 0x801b00ebu:
		return __tsc(u8"{Display Driver Stopped Responding and recovered} The %hs display driver has stopped working normally. The recovery had been performed.");
	case 0x801c0001u:
		return __tsc(u8"{Buffer too small} The buffer is too small to contain the entry. No information has been written to the buffer.");
	case 0x801e0000u:
		return __tsc(u8"");
	case 0x80210001u:
		return __tsc(u8"Volume metadata read or write is incomplete.");
	case 0x80210002u:
		return __tsc(u8"BitLocker encryption keys were ignored because the volume was in a transient state.");
	case 0x80370001u:
	case 0x80380001u:
	case 0x80380002u:
	case 0x80390001u:
	case 0x80390003u:
	case 0x803a0001u:
	case 0x803f0001u:
	case 0x80430006u:
		return __tsc(u8"");
	case 0xc0000001u:
		return __tsc(u8"{Operation Failed}\nThe requested operation was unsuccessful.");
	case 0xc0000002u:
		return __tsc(u8"{Not Implemented}\nThe requested operation is not implemented.");
	case 0xc0000003u:
		return __tsc(u8"{Invalid Parameter}\nThe specified information class is not a valid information class for the specified object.");
	case 0xc0000004u:
		return __tsc(u8"The specified information record length does not match the length required for the specified information class.");
	case 0xc0000005u:
		return __tsc(u8"The instruction at 0x%p referenced memory at 0x%p. The memory could not be %s.");
	case 0xc0000006u:
		return __tsc(u8"The instruction at 0x%p referenced memory at 0x%p. The required data was not placed into memory because of an I/O error status of 0x%x.");
	case 0xc0000007u:
		return __tsc(u8"The pagefile quota for the process has been exhausted.");
	case 0xc0000008u:
		return __tsc(u8"An invalid HANDLE was specified.");
	case 0xc0000009u:
		return __tsc(u8"An invalid initial stack was specified in a call to NtCreateThread.");
	case 0xc000000au:
		return __tsc(u8"An invalid initial start address was specified in a call to NtCreateThread.");
	case 0xc000000bu:
		return __tsc(u8"An invalid Client ID was specified.");
	case 0xc000000cu:
		return __tsc(u8"An attempt was made to cancel or set a timer that has an associated APC and the subject thread is not the thread that originally set the timer with an associated APC routine.");
	case 0xc000000du:
		return __tsc(u8"An invalid parameter was passed to a service or function.");
	case 0xc000000eu:
		return __tsc(u8"A device which does not exist was specified.");
	case 0xc000000fu:
		return __tsc(u8"{File Not Found}\nThe file %hs does not exist.");
	case 0xc0000010u:
		return __tsc(u8"The specified request is not a valid operation for the target device.");
	case 0xc0000011u:
		return __tsc(u8"The end-of-file marker has been reached. There is no valid data in the file beyond this marker.");
	case 0xc0000012u:
		return __tsc(u8"{Wrong Volume}\nThe wrong volume is in the drive.\nPlease insert volume %hs into drive %hs.");
	case 0xc0000013u:
		return __tsc(u8"{No Disk}\nThere is no disk in the drive.\nPlease insert a disk into drive %hs.");
	case 0xc0000014u:
		return __tsc(u8"{Unknown Disk Format}\nThe disk in drive %hs is not formatted properly.\nPlease check the disk, and reformat if necessary.");
	case 0xc0000015u:
		return __tsc(u8"{Sector Not Found}\nThe specified sector does not exist.");
	case 0xc0000016u:
		return __tsc(u8"{Still Busy}\nThe specified I/O request packet (IRP) cannot be disposed of because the I/O operation is not complete.");
	case 0xc0000017u:
		return __tsc(u8"{Not Enough Quota}\nNot enough virtual memory or paging file quota is available to complete the specified operation.");
	case 0xc0000018u:
		return __tsc(u8"{Conflicting Address Range}\nThe specified address range conflicts with the address space.");
	case 0xc0000019u:
		return __tsc(u8"Address range to unmap is not a mapped view.");
	case 0xc000001au:
		return __tsc(u8"Virtual memory cannot be freed.");
	case 0xc000001bu:
		return __tsc(u8"Specified section cannot be deleted.");
	case 0xc000001cu:
		return __tsc(u8"An invalid system service was specified in a system service call.");
	case 0xc000001du:
		return __tsc(u8"{EXCEPTION}\nIllegal Instruction\nAn attempt was made to execute an illegal instruction.");
	case 0xc000001eu:
		return __tsc(u8"{Invalid Lock Sequence}\nAn attempt was made to execute an invalid lock sequence.");
	case 0xc000001fu:
		return __tsc(u8"{Invalid Mapping}\nAn attempt was made to create a view for a section which is bigger than the section.");
	case 0xc0000020u:
		return __tsc(u8"{Bad File}\nThe attributes of the specified mapping file for a section of memory cannot be read.");
	case 0xc0000021u:
		return __tsc(u8"{Already Committed}\nThe specified address range is already committed.");
	case 0xc0000022u:
		return __tsc(u8"{Access Denied}\nA process has requested access to an object, but has not been granted those access rights.");
	case 0xc0000023u:
		return __tsc(u8"{Buffer Too Small}\nThe buffer is too small to contain the entry. No information has been written to the buffer.");
	case 0xc0000024u:
		return __tsc(u8"{Wrong Type}\nThere is a mismatch between the type of object required by the requested operation and the type of object that is specified in the request.");
	case 0xc0000025u:
		return __tsc(u8"{EXCEPTION}\nCannot Continue\nWindows cannot continue from this exception.");
	case 0xc0000026u:
		return __tsc(u8"An invalid exception disposition was returned by an exception handler.");
	case 0xc0000027u:
		return __tsc(u8"Unwind exception code.");
	case 0xc0000028u:
		return __tsc(u8"An invalid or unaligned stack was encountered during an unwind operation.");
	case 0xc0000029u:
		return __tsc(u8"An invalid unwind target was encountered during an unwind operation.");
	case 0xc000002au:
		return __tsc(u8"An attempt was made to unlock a page of memory which was not locked.");
	case 0xc000002bu:
		return __tsc(u8"Device parity error on I/O operation.");
	case 0xc000002cu:
		return __tsc(u8"An attempt was made to decommit uncommitted virtual memory.");
	case 0xc000002du:
		return __tsc(u8"An attempt was made to change the attributes on memory that has not been committed.");
	case 0xc000002eu:
		return __tsc(u8"Invalid Object Attributes specified to NtCreatePort or invalid Port Attributes specified to NtConnectPort");
	case 0xc000002fu:
		return __tsc(u8"Length of message passed to NtRequestPort or NtRequestWaitReplyPort was longer than the maximum message allowed by the port.");
	case 0xc0000030u:
		return __tsc(u8"An invalid combination of parameters was specified.");
	case 0xc0000031u:
		return __tsc(u8"An attempt was made to lower a quota limit below the current usage.");
	case 0xc0000032u:
		return __tsc(u8"{Corrupt Disk}\nThe file system structure on the disk is corrupt and unusable.\nPlease run the Chkdsk utility on the volume %hs.");
	case 0xc0000033u:
		return __tsc(u8"Object Name invalid.");
	case 0xc0000034u:
		return __tsc(u8"Object Name not found.");
	case 0xc0000035u:
		return __tsc(u8"Object Name already exists.");
	case 0xc0000036u:
		return __tsc(u8"A port with the 'do not disturb' flag set attempted to send a message to a port in a suspended process.\nThe process was not woken, and the message was not delivered.");
	case 0xc0000037u:
		return __tsc(u8"Attempt to send a message to a disconnected communication port.");
	case 0xc0000038u:
		return __tsc(u8"An attempt was made to attach to a device that was already attached to another device.");
	case 0xc0000039u:
		return __tsc(u8"Object Path Component was not a directory object.");
	case 0xc000003au:
		return __tsc(u8"{Path Not Found}\nThe path %hs does not exist.");
	case 0xc000003bu:
		return __tsc(u8"Object Path Component was not a directory object.");
	case 0xc000003cu:
		return __tsc(u8"{Data Overrun}\nA data overrun error occurred.");
	case 0xc000003du:
		return __tsc(u8"{Data Late}\nA data late error occurred.");
	case 0xc000003eu:
		return __tsc(u8"{Data Error}\nAn error in reading or writing data occurred.");
	case 0xc000003fu:
		return __tsc(u8"{Bad CRC}\nA cyclic redundancy check (CRC) checksum error occurred.");
	case 0xc0000040u:
		return __tsc(u8"{Section Too Large}\nThe specified section is too big to map the file.");
	case 0xc0000041u:
		return __tsc(u8"The NtConnectPort request is refused.");
	case 0xc0000042u:
		return __tsc(u8"The type of port handle is invalid for the operation requested.");
	case 0xc0000043u:
		return __tsc(u8"A file cannot be opened because the share access flags are incompatible.");
	case 0xc0000044u:
		return __tsc(u8"Insufficient quota exists to complete the operation");
	case 0xc0000045u:
		return __tsc(u8"The specified page protection was not valid.");
	case 0xc0000046u:
		return __tsc(u8"An attempt to release a mutant object was made by a thread that was not the owner of the mutant object.");
	case 0xc0000047u:
		return __tsc(u8"An attempt was made to release a semaphore such that its maximum count would have been exceeded.");
	case 0xc0000048u:
		return __tsc(u8"An attempt to set a process's DebugPort or ExceptionPort was made, but a port already exists in the process or an attempt to set a file's CompletionPort made, but a port was already set in the file or an attempt to set an ALPC port's associated completion port was made, but it is already set.");
	case 0xc0000049u:
		return __tsc(u8"An attempt was made to query image information on a section which does not map an image.");
	case 0xc000004au:
		return __tsc(u8"An attempt was made to suspend a thread whose suspend count was at its maximum.");
	case 0xc000004bu:
		return __tsc(u8"An attempt was made to access a thread that has begun termination.");
	case 0xc000004cu:
		return __tsc(u8"An attempt was made to set the working set limit to an invalid value (minimum greater than maximum, etc).");
	case 0xc000004du:
		return __tsc(u8"A section was created to map a file which is not compatible to an already existing section which maps the same file.");
	case 0xc000004eu:
		return __tsc(u8"A view to a section specifies a protection which is incompatible with the initial view's protection.");
	case 0xc000004fu:
		return __tsc(u8"An operation involving EAs failed because the file system does not support EAs.");
	case 0xc0000050u:
		return __tsc(u8"An EA operation failed because EA set is too large.");
	case 0xc0000051u:
		return __tsc(u8"An EA operation failed because the name or EA index is invalid.");
	case 0xc0000052u:
		return __tsc(u8"The file for which EAs were requested has no EAs.");
	case 0xc0000053u:
		return __tsc(u8"The EA is corrupt and non-readable.");
	case 0xc0000054u:
		return __tsc(u8"A requested read/write cannot be granted due to a conflicting file lock.");
	case 0xc0000055u:
		return __tsc(u8"A requested file lock cannot be granted due to other existing locks.");
	case 0xc0000056u:
		return __tsc(u8"A non close operation has been requested of a file object with a delete pending.");
	case 0xc0000057u:
		return __tsc(u8"An attempt was made to set the control attribute on a file. This attribute is not supported in the target file system.");
	case 0xc0000058u:
		return __tsc(u8"Indicates a revision number encountered or specified is not one known by the service. It may be a more recent revision than the service is aware of.");
	case 0xc0000059u:
		return __tsc(u8"Indicates two revision levels are incompatible.");
	case 0xc000005au:
		return __tsc(u8"Indicates a particular Security ID may not be assigned as the owner of an object.");
	case 0xc000005bu:
		return __tsc(u8"Indicates a particular Security ID may not be assigned as the primary group of an object.");
	case 0xc000005cu:
		return __tsc(u8"An attempt has been made to operate on an impersonation token by a thread that is not currently impersonating a client.");
	case 0xc000005du:
		return __tsc(u8"A mandatory group may not be disabled.");
	case 0xc000005eu:
		return __tsc(u8"We can't sign you in with this credential because your domain isn't available. Make sure your device is connected to your organization's network and try again. If you previously signed in on this device with another credential, you can sign in with that credential.");
	case 0xc000005fu:
		return __tsc(u8"A specified logon session does not exist. It may already have been terminated.");
	case 0xc0000060u:
		return __tsc(u8"A specified privilege does not exist.");
	case 0xc0000061u:
		return __tsc(u8"A required privilege is not held by the client.");
	case 0xc0000062u:
		return __tsc(u8"The name provided is not a properly formed account name.");
	case 0xc0000063u:
		return __tsc(u8"The specified account already exists.");
	case 0xc0000064u:
		return __tsc(u8"The specified account does not exist.");
	case 0xc0000065u:
		return __tsc(u8"The specified group already exists.");
	case 0xc0000066u:
		return __tsc(u8"The specified group does not exist.");
	case 0xc0000067u:
		return __tsc(u8"The specified user account is already in the specified group account. Also used to indicate a group cannot be deleted because it contains a member.");
	case 0xc0000068u:
		return __tsc(u8"The specified user account is not a member of the specified group account.");
	case 0xc0000069u:
		return __tsc(u8"Indicates the requested operation would disable, delete or could prevent logon for an administration account.\nThis is not allowed to prevent creating a situation in which the system cannot be administrated.");
	case 0xc000006au:
		return __tsc(u8"When trying to update a password, this return status indicates that the value provided as the current password is not correct.");
	case 0xc000006bu:
		return __tsc(u8"When trying to update a password, this return status indicates that the value provided for the new password contains values that are not allowed in passwords.");
	case 0xc000006cu:
		return __tsc(u8"When trying to update a password, this status indicates that some password update rule has been violated. For example, the password may not meet length criteria.");
	case 0xc000006du:
		return __tsc(u8"The attempted logon is invalid. This is either due to a bad username or authentication information.");
	case 0xc000006eu:
		return __tsc(u8"Indicates a referenced user name and authentication information are valid, but some user account restriction has prevented successful authentication (such as time-of-day restrictions).");
	case 0xc000006fu:
		return __tsc(u8"The user account has time restrictions and may not be logged onto at this time.");
	case 0xc0000070u:
		return __tsc(u8"The user account is restricted such that it may not be used to log on from the source workstation.");
	case 0xc0000071u:
		return __tsc(u8"The user account's password has expired.");
	case 0xc0000072u:
		return __tsc(u8"The referenced account is currently disabled and may not be logged on to.");
	case 0xc0000073u:
		return __tsc(u8"None of the information to be translated has been translated.");
	case 0xc0000074u:
		return __tsc(u8"The number of LUIDs requested may not be allocated with a single allocation.");
	case 0xc0000075u:
		return __tsc(u8"Indicates there are no more LUIDs to allocate.");
	case 0xc0000076u:
		return __tsc(u8"Indicates the sub-authority value is invalid for the particular use.");
	case 0xc0000077u:
		return __tsc(u8"Indicates the ACL structure is not valid.");
	case 0xc0000078u:
		return __tsc(u8"Indicates the SID structure is not valid.");
	case 0xc0000079u:
		return __tsc(u8"Indicates the SECURITY_DESCRIPTOR structure is not valid.");
	case 0xc000007au:
		return __tsc(u8"Indicates the specified procedure address cannot be found in the DLL.");
	case 0xc000007bu:
		return __tsc(u8"{Bad Image}\n%hs is either not designed to run on Windows or it contains an error. Try installing the program again using the original installation media or contact your system administrator or the software vendor for support. Error status 0x");
	case 0xc000007cu:
		return __tsc(u8"An attempt was made to reference a token that doesn't exist.\nThis is typically done by referencing the token associated with a thread when the thread is not impersonating a client.");
	case 0xc000007du:
		return __tsc(u8"Indicates that an attempt to build either an inherited ACL or ACE was not successful.\nThis can be caused by a number of things. One of the more probable causes is the replacement of a CreatorId with an SID that didn't fit into the ACE or ACL.");
	case 0xc000007eu:
		return __tsc(u8"The range specified in NtUnlockFile was not locked.");
	case 0xc000007fu:
		return __tsc(u8"An operation failed because the disk was full.\nIf this is a thinly provisioned volume the physical storage backing this volume has been exhausted.");
	case 0xc0000080u:
		return __tsc(u8"The GUID allocation server is [already] disabled at the moment.");
	case 0xc0000081u:
		return __tsc(u8"The GUID allocation server is [already] enabled at the moment.");
	case 0xc0000082u:
		return __tsc(u8"Too many GUIDs were requested from the allocation server at once.");
	case 0xc0000083u:
		return __tsc(u8"The GUIDs could not be allocated because the Authority Agent was exhausted.");
	case 0xc0000084u:
		return __tsc(u8"The value provided was an invalid value for an identifier authority.");
	case 0xc0000085u:
		return __tsc(u8"There are no more authority agent values available for the given identifier authority value.");
	case 0xc0000086u:
		return __tsc(u8"An invalid volume label has been specified.");
	case 0xc0000087u:
		return __tsc(u8"A mapped section could not be extended.");
	case 0xc0000088u:
		return __tsc(u8"Specified section to flush does not map a data file.");
	case 0xc0000089u:
		return __tsc(u8"Indicates the specified image file did not contain a resource section.");
	case 0xc000008au:
		return __tsc(u8"Indicates the specified resource type cannot be found in the image file.");
	case 0xc000008bu:
		return __tsc(u8"Indicates the specified resource name cannot be found in the image file.");
	case 0xc000008cu:
		return __tsc(u8"{EXCEPTION}\nArray bounds exceeded.");
	case 0xc000008du:
		return __tsc(u8"{EXCEPTION}\nFloating-point denormal operand.");
	case 0xc000008eu:
		return __tsc(u8"{EXCEPTION}\nFloating-point division by zero.");
	case 0xc000008fu:
		return __tsc(u8"{EXCEPTION}\nFloating-point inexact result.");
	case 0xc0000090u:
		return __tsc(u8"{EXCEPTION}\nFloating-point invalid operation.");
	case 0xc0000091u:
		return __tsc(u8"{EXCEPTION}\nFloating-point overflow.");
	case 0xc0000092u:
		return __tsc(u8"{EXCEPTION}\nFloating-point stack check.");
	case 0xc0000093u:
		return __tsc(u8"{EXCEPTION}\nFloating-point underflow.");
	case 0xc0000094u:
		return __tsc(u8"{EXCEPTION}\nInteger division by zero.");
	case 0xc0000095u:
		return __tsc(u8"{EXCEPTION}\nInteger overflow.");
	case 0xc0000096u:
		return __tsc(u8"{EXCEPTION}\nPrivileged instruction.");
	case 0xc0000097u:
		return __tsc(u8"An attempt was made to install more paging files than the system supports.");
	case 0xc0000098u:
		return __tsc(u8"The volume for a file has been externally altered such that the opened file is no longer valid.");
	case 0xc0000099u:
		return __tsc(u8"When a block of memory is allotted for future updates, such as the memory allocated to hold discretionary access control and primary group information, successive updates may exceed the amount of memory originally allotted.\nSince quota may already have been charged to several processes which have handles to the object, it is not reasonable to alter the size of the allocated memory.\nInstead, a request that requires more memory than has been allotted must fail and the STATUS_ALLOTED_SPACE_EXCEEDED error returned.");
	case 0xc000009au:
		return __tsc(u8"Insufficient system resources exist to complete the API.");
	case 0xc000009bu:
		return __tsc(u8"An attempt has been made to open a DFS exit path control file.");
	case 0xc000009cu:
		return __tsc(u8"STATUS_DEVICE_DATA_ERROR");
	case 0xc000009du:
		return __tsc(u8"STATUS_DEVICE_NOT_CONNECTED");
	case 0xc000009eu:
		return __tsc(u8"STATUS_DEVICE_POWER_FAILURE");
	case 0xc000009fu:
		return __tsc(u8"Virtual memory cannot be freed as base address is not the base of the region and a region size of zero was specified.");
	case 0xc00000a0u:
		return __tsc(u8"An attempt was made to free virtual memory which is not allocated.");
	case 0xc00000a1u:
		return __tsc(u8"The working set is not big enough to allow the requested pages to be locked.");
	case 0xc00000a2u:
		return __tsc(u8"{Write Protect Error}\nThe disk cannot be written to because it is write protected. Please remove the write protection from the volume %hs in drive %hs.");
	case 0xc00000a3u:
		return __tsc(u8"{Drive Not Ready}\nThe drive is not ready for use; its door may be open. Please check drive %hs and make sure that a disk is inserted and that the drive door is closed.");
	case 0xc00000a4u:
		return __tsc(u8"The specified attributes are invalid, or incompatible with the attributes for the group as a whole.");
	case 0xc00000a5u:
		return __tsc(u8"A specified impersonation level is invalid.\nAlso used to indicate a required impersonation level was not provided.");
	case 0xc00000a6u:
		return __tsc(u8"An attempt was made to open an Anonymous level token.\nAnonymous tokens may not be opened.");
	case 0xc00000a7u:
		return __tsc(u8"The validation information class requested was invalid.");
	case 0xc00000a8u:
	case 0xc00000a9u:
		return __tsc(u8"The type of a token object is inappropriate for its attempted use.");
	case 0xc00000aau:
		return __tsc(u8"An attempt was made to execute an instruction at an unaligned address and the host system does not support unaligned instruction references.");
	case 0xc00000abu:
		return __tsc(u8"The maximum named pipe instance count has been reached.");
	case 0xc00000acu:
		return __tsc(u8"An instance of a named pipe cannot be found in the listening state.");
	case 0xc00000adu:
		return __tsc(u8"The named pipe is not in the connected or closing state.");
	case 0xc00000aeu:
		return __tsc(u8"The specified pipe is set to complete operations and there are current I/O operations queued so it cannot be changed to queue operations.");
	case 0xc00000afu:
		return __tsc(u8"The specified handle is not open to the server end of the named pipe.");
	case 0xc00000b0u:
		return __tsc(u8"The specified named pipe is in the disconnected state.");
	case 0xc00000b1u:
		return __tsc(u8"The specified named pipe is in the closing state.");
	case 0xc00000b2u:
		return __tsc(u8"The specified named pipe is in the connected state.");
	case 0xc00000b3u:
		return __tsc(u8"The specified named pipe is in the listening state.");
	case 0xc00000b4u:
		return __tsc(u8"The specified named pipe is not in message mode.");
	case 0xc00000b5u:
		return __tsc(u8"{Device Timeout}\nThe specified I/O operation on %hs was not completed before the time-out period expired.");
	case 0xc00000b6u:
		return __tsc(u8"The specified file has been closed by another process.");
	case 0xc00000b7u:
		return __tsc(u8"Profiling not started.");
	case 0xc00000b8u:
		return __tsc(u8"Profiling not stopped.");
	case 0xc00000b9u:
		return __tsc(u8"The passed ACL did not contain the minimum required information.");
	case 0xc00000bau:
		return __tsc(u8"The file that was specified as a target is a directory and the caller specified that it could be anything but a directory.");
	case 0xc00000bbu:
		return __tsc(u8"The request is not supported.");
	case 0xc00000bcu:
		return __tsc(u8"This remote computer is not listening.");
	case 0xc00000bdu:
		return __tsc(u8"A duplicate name exists on the network.");
	case 0xc00000beu:
		return __tsc(u8"The network path cannot be located.");
	case 0xc00000bfu:
		return __tsc(u8"The network is busy.");
	case 0xc00000c0u:
		return __tsc(u8"This device does not exist.");
	case 0xc00000c1u:
		return __tsc(u8"The network BIOS command limit has been reached.");
	case 0xc00000c2u:
		return __tsc(u8"An I/O adapter hardware error has occurred.");
	case 0xc00000c3u:
		return __tsc(u8"The network responded incorrectly.");
	case 0xc00000c4u:
		return __tsc(u8"An unexpected network error occurred.");
	case 0xc00000c5u:
		return __tsc(u8"The remote adapter is not compatible.");
	case 0xc00000c6u:
		return __tsc(u8"The printer queue is full.");
	case 0xc00000c7u:
		return __tsc(u8"Space to store the file waiting to be printed is not available on the server.");
	case 0xc00000c8u:
		return __tsc(u8"The requested print file has been canceled.");
	case 0xc00000c9u:
		return __tsc(u8"The network name was deleted.");
	case 0xc00000cau:
		return __tsc(u8"Network access is denied.");
	case 0xc00000cbu:
		return __tsc(u8"{Incorrect Network Resource Type}\nThe specified device type (LPT, for example) conflicts with the actual device type on the remote resource.");
	case 0xc00000ccu:
		return __tsc(u8"{Network Name Not Found}\nThe specified share name cannot be found on the remote server.");
	case 0xc00000cdu:
		return __tsc(u8"The name limit for the local computer network adapter card was exceeded.");
	case 0xc00000ceu:
		return __tsc(u8"The network BIOS session limit was exceeded.");
	case 0xc00000cfu:
		return __tsc(u8"File sharing has been temporarily paused.");
	case 0xc00000d0u:
		return __tsc(u8"No more connections can be made to this remote computer at this time because there are already as many connections as the computer can accept.");
	case 0xc00000d1u:
		return __tsc(u8"Print or disk redirection is temporarily paused.");
	case 0xc00000d2u:
		return __tsc(u8"A network data fault occurred.");
	case 0xc00000d3u:
		return __tsc(u8"The number of active profiling objects is at the maximum and no more may be started.");
	case 0xc00000d4u:
		return __tsc(u8"{Incorrect Volume}\nThe target file of a rename request is located on a different device than the source of the rename request.");
	case 0xc00000d5u:
		return __tsc(u8"The file specified has been renamed and thus cannot be modified.");
	case 0xc00000d6u:
		return __tsc(u8"{Network Request Timeout}\nThe session with a remote server has been disconnected because the time-out interval for a request has expired.");
	case 0xc00000d7u:
		return __tsc(u8"Indicates an attempt was made to operate on the security of an object that does not have security associated with it.");
	case 0xc00000d8u:
		return __tsc(u8"Used to indicate that an operation cannot continue without blocking for I/O.");
	case 0xc00000d9u:
		return __tsc(u8"Used to indicate that a read operation was done on an empty pipe.");
	case 0xc00000dau:
		return __tsc(u8"Configuration information could not be read from the domain controller, either because the machine is unavailable, or access has been denied.");
	case 0xc00000dbu:
		return __tsc(u8"Indicates that a thread attempted to terminate itself by default (called NtTerminateThread with NULL) and it was the last thread in the current process.");
	case 0xc00000dcu:
		return __tsc(u8"Indicates the Sam Server was in the wrong state to perform the desired operation.");
	case 0xc00000ddu:
		return __tsc(u8"Indicates the Domain was in the wrong state to perform the desired operation.");
	case 0xc00000deu:
		return __tsc(u8"This operation is only allowed for the Primary Domain Controller of the domain.");
	case 0xc00000dfu:
		return __tsc(u8"The specified Domain did not exist.");
	case 0xc00000e0u:
		return __tsc(u8"The specified Domain already exists.");
	case 0xc00000e1u:
		return __tsc(u8"An attempt was made to exceed the limit on the number of domains per server for this release.");
	case 0xc00000e2u:
		return __tsc(u8"Error status returned when oplock request is denied.");
	case 0xc00000e3u:
		return __tsc(u8"Error status returned when an invalid oplock acknowledgment is received by a file system.");
	case 0xc00000e4u:
		return __tsc(u8"This error indicates that the requested operation cannot be completed due to a catastrophic media failure or on-disk data structure corruption.");
	case 0xc00000e5u:
		return __tsc(u8"An internal error occurred.");
	case 0xc00000e6u:
		return __tsc(u8"Indicates generic access types were contained in an access mask which should already be mapped to non-generic access types.");
	case 0xc00000e7u:
		return __tsc(u8"Indicates a security descriptor is not in the necessary format (absolute or self-relative).");
	case 0xc00000e8u:
		return __tsc(u8"An access to a user buffer failed at an \"expected\" point in time. This code is defined since the caller does not want to accept STATUS_ACCESS_VIOLATION in its filter.");
	case 0xc00000e9u:
		return __tsc(u8"If an I/O error is returned which is not defined in the standard FsRtl filter, it is converted to the following error which is guaranteed to be in the filter. In this case information is lost, however, the filter correctly handles the exception.");
	case 0xc00000eau:
	case 0xc00000ebu:
	case 0xc00000ecu:
		return __tsc(u8"If an MM error is returned which is not defined in the standard FsRtl filter, it is converted to one of the following errors which is guaranteed to be in the filter. In this case information is lost, however, the filter correctly handles the exception.");
	case 0xc00000edu:
		return __tsc(u8"The requested action is restricted for use by logon processes only. The calling process has not registered as a logon process.");
	case 0xc00000eeu:
		return __tsc(u8"An attempt has been made to start a new session manager or LSA logon session with an ID that is already in use.");
	case 0xc00000efu:
		return __tsc(u8"An invalid parameter was passed to a service or function as the first argument.");
	case 0xc00000f0u:
		return __tsc(u8"An invalid parameter was passed to a service or function as the second argument.");
	case 0xc00000f1u:
		return __tsc(u8"An invalid parameter was passed to a service or function as the third argument.");
	case 0xc00000f2u:
		return __tsc(u8"An invalid parameter was passed to a service or function as the fourth argument.");
	case 0xc00000f3u:
		return __tsc(u8"An invalid parameter was passed to a service or function as the fifth argument.");
	case 0xc00000f4u:
		return __tsc(u8"An invalid parameter was passed to a service or function as the sixth argument.");
	case 0xc00000f5u:
		return __tsc(u8"An invalid parameter was passed to a service or function as the seventh argument.");
	case 0xc00000f6u:
		return __tsc(u8"An invalid parameter was passed to a service or function as the eighth argument.");
	case 0xc00000f7u:
		return __tsc(u8"An invalid parameter was passed to a service or function as the ninth argument.");
	case 0xc00000f8u:
		return __tsc(u8"An invalid parameter was passed to a service or function as the tenth argument.");
	case 0xc00000f9u:
		return __tsc(u8"An invalid parameter was passed to a service or function as the eleventh argument.");
	case 0xc00000fau:
		return __tsc(u8"An invalid parameter was passed to a service or function as the twelfth argument.");
	case 0xc00000fbu:
		return __tsc(u8"An attempt was made to access a network file, but the network software was not yet started.");
	case 0xc00000fcu:
		return __tsc(u8"An attempt was made to start the redirector, but the redirector has already been started.");
	case 0xc00000fdu:
		return __tsc(u8"A new guard page for the stack cannot be created.");
	case 0xc00000feu:
		return __tsc(u8"A specified authentication package is unknown.");
	case 0xc00000ffu:
		return __tsc(u8"A malformed function table was encountered during an unwind operation.");
	case 0xc0000100u:
		return __tsc(u8"Indicates the specified environment variable name was not found in the specified environment block.");
	case 0xc0000101u:
		return __tsc(u8"Indicates that the directory trying to be deleted is not empty.");
	case 0xc0000102u:
		return __tsc(u8"{Corrupt File}\nThe file or directory %hs is corrupt and unreadable.\nPlease run the Chkdsk utility.");
	case 0xc0000103u:
		return __tsc(u8"A requested opened file is not a directory.");
	case 0xc0000104u:
		return __tsc(u8"The logon session is not in a state that is consistent with the requested operation.");
	case 0xc0000105u:
		return __tsc(u8"An internal LSA error has occurred. An authentication package has requested the creation of a Logon Session but the ID of an already existing Logon Session has been specified.");
	case 0xc0000106u:
		return __tsc(u8"A specified name string is too long for its intended use.");
	case 0xc0000107u:
		return __tsc(u8"The user attempted to force close the files on a redirected drive, but there were opened files on the drive, and the user did not specify a sufficient level of force.");
	case 0xc0000108u:
		return __tsc(u8"The user attempted to force close the files on a redirected drive, but there were opened directories on the drive, and the user did not specify a sufficient level of force.");
	case 0xc0000109u:
		return __tsc(u8"RtlFindMessage could not locate the requested message ID in the message table resource.");
	case 0xc000010au:
		return __tsc(u8"An attempt was made to access an exiting process.");
	case 0xc000010bu:
		return __tsc(u8"Indicates an invalid value has been provided for the LogonType requested.");
	case 0xc000010cu:
		return __tsc(u8"Indicates that an attempt was made to assign protection to a file system file or directory and one of the SIDs in the security descriptor could not be translated into a GUID that could be stored by the file system.\nThis causes the protection attempt to fail, which may cause a file creation attempt to fail.");
	case 0xc000010du:
		return __tsc(u8"Indicates that an attempt has been made to impersonate via a named pipe that has not yet been read from.");
	case 0xc000010eu:
		return __tsc(u8"Indicates that the specified image is already loaded.");
	case 0xc000010fu:
		return __tsc(u8"STATUS_ABIOS_NOT_PRESENT");
	case 0xc0000110u:
		return __tsc(u8"STATUS_ABIOS_LID_NOT_EXIST");
	case 0xc0000111u:
		return __tsc(u8"STATUS_ABIOS_LID_ALREADY_OWNED");
	case 0xc0000112u:
		return __tsc(u8"STATUS_ABIOS_NOT_LID_OWNER");
	case 0xc0000113u:
		return __tsc(u8"STATUS_ABIOS_INVALID_COMMAND");
	case 0xc0000114u:
		return __tsc(u8"STATUS_ABIOS_INVALID_LID");
	case 0xc0000115u:
		return __tsc(u8"STATUS_ABIOS_SELECTOR_NOT_AVAILABLE");
	case 0xc0000116u:
		return __tsc(u8"STATUS_ABIOS_INVALID_SELECTOR");
	case 0xc0000117u:
		return __tsc(u8"Indicates that an attempt was made to change the size of the LDT for a process that has no LDT.");
	case 0xc0000118u:
		return __tsc(u8"Indicates that an attempt was made to grow an LDT by setting its size, or that the size was not an even number of selectors.");
	case 0xc0000119u:
		return __tsc(u8"Indicates that the starting value for the LDT information was not an integral multiple of the selector size.");
	case 0xc000011au:
		return __tsc(u8"Indicates that the user supplied an invalid descriptor when trying to set up Ldt descriptors.");
	case 0xc000011bu:
		return __tsc(u8"The specified image file did not have the correct format. It appears to be NE format.");
	case 0xc000011cu:
		return __tsc(u8"Indicates that the transaction state of a registry sub-tree is incompatible with the requested operation. For example, a request has been made to start a new transaction with one already in progress, or a request has been made to apply a transaction when one is not currently in progress.");
	case 0xc000011du:
		return __tsc(u8"Indicates an error has occurred during a registry transaction commit. The database has been left in an unknown, but probably inconsistent, state. The state of the registry transaction is left as COMMITTING.");
	case 0xc000011eu:
		return __tsc(u8"An attempt was made to map a file of size zero with the maximum size specified as zero.");
	case 0xc000011fu:
		return __tsc(u8"Too many files are opened on a remote server.\nThis error should only be returned by the Windows redirector on a remote drive.");
	case 0xc0000120u:
		return __tsc(u8"The I/O request was canceled.");
	case 0xc0000121u:
		return __tsc(u8"An attempt has been made to remove a file or directory that cannot be deleted.");
	case 0xc0000122u:
		return __tsc(u8"Indicates a name specified as a remote computer name is syntactically invalid.");
	case 0xc0000123u:
		return __tsc(u8"An I/O request other than close was performed on a file after it has been deleted, which can only happen to a request which did not complete before the last handle was closed via NtClose.");
	case 0xc0000124u:
		return __tsc(u8"Indicates an operation has been attempted on a built-in (special) SAM account which is incompatible with built-in accounts. For example, built-in accounts cannot be deleted.");
	case 0xc0000125u:
		return __tsc(u8"The operation requested may not be performed on the specified group because it is a built-in special group.");
	case 0xc0000126u:
		return __tsc(u8"The operation requested may not be performed on the specified user because it is a built-in special user.");
	case 0xc0000127u:
		return __tsc(u8"Indicates a member cannot be removed from a group because the group is currently the member's primary group.");
	case 0xc0000128u:
		return __tsc(u8"An I/O request other than close and several other special case operations was attempted using a file object that had already been closed.");
	case 0xc0000129u:
		return __tsc(u8"Indicates a process has too many threads to perform the requested action. For example, assignment of a primary token may only be performed when a process has zero or one threads.");
	case 0xc000012au:
		return __tsc(u8"An attempt was made to operate on a thread within a specific process, but the thread specified is not in the process specified.");
	case 0xc000012bu:
		return __tsc(u8"An attempt was made to establish a token for use as a primary token but the token is already in use. A token can only be the primary token of one process at a time.");
	case 0xc000012cu:
		return __tsc(u8"Page file quota was exceeded.");
	case 0xc000012du:
		return __tsc(u8"{Out of Virtual Memory}\nYour system is low on virtual memory. To ensure that Windows runs properly, increase the size of your virtual memory paging file. For more information, see Help.");
	case 0xc000012eu:
		return __tsc(u8"The specified image file did not have the correct format, it appears to be LE format.");
	case 0xc000012fu:
		return __tsc(u8"The specified image file did not have the correct format, it did not have an initial MZ.");
	case 0xc0000130u:
		return __tsc(u8"The specified image file did not have the correct format, it did not have a proper e_lfarlc in the MZ header.");
	case 0xc0000131u:
		return __tsc(u8"The specified image file did not have the correct format, it appears to be a 16-bit Windows image.");
	case 0xc0000132u:
		return __tsc(u8"The Netlogon service cannot start because another Netlogon service running in the domain conflicts with the specified role.");
	case 0xc0000133u:
		return __tsc(u8"The time at the Primary Domain Controller is different than the time at the Backup Domain Controller or member server by too large an amount.");
	case 0xc0000134u:
		return __tsc(u8"The SAM database on a Windows Server is significantly out of synchronization with the copy on the Domain Controller. A complete synchronization is required.");
	case 0xc0000135u:
		return __tsc(u8"The code execution cannot proceed because %hs was not found. Reinstalling the program may fix this problem.");
	case 0xc0000136u:
		return __tsc(u8"The NtCreateFile API failed. This error should never be returned to an application, it is a place holder for the Windows Lan Manager Redirector to use in its internal error mapping routines.");
	case 0xc0000137u:
		return __tsc(u8"{Privilege Failed}\nThe I/O permissions for the process could not be changed.");
	case 0xc0000138u:
		return __tsc(u8"{Ordinal Not Found}\nThe ordinal %ld could not be located in the dynamic link library %hs.");
	case 0xc0000139u:
		return __tsc(u8"{Entry Point Not Found}\nThe procedure entry point %hs could not be located in the dynamic link library %hs.");
	case 0xc000013au:
		return __tsc(u8"{Application Exit by CTRL+C}\nThe application terminated as a result of a CTRL+C.");
	case 0xc000013bu:
		return __tsc(u8"{Virtual Circuit Closed}\nThe network transport on your computer has closed a network connection. There may or may not be I/O requests outstanding.");
	case 0xc000013cu:
		return __tsc(u8"{Virtual Circuit Closed}\nThe network transport on a remote computer has closed a network connection. There may or may not be I/O requests outstanding.");
	case 0xc000013du:
		return __tsc(u8"{Insufficient Resources on Remote Computer}\nThe remote computer has insufficient resources to complete the network request. For instance, there may not be enough memory available on the remote computer to carry out the request at this time.");
	case 0xc000013eu:
		return __tsc(u8"{Virtual Circuit Closed}\nAn existing connection (virtual circuit) has been broken at the remote computer. There is probably something wrong with the network software protocol or the network hardware on the remote computer.");
	case 0xc000013fu:
		return __tsc(u8"{Virtual Circuit Closed}\nThe network transport on your computer has closed a network connection because it had to wait too long for a response from the remote computer.");
	case 0xc0000140u:
		return __tsc(u8"The connection handle given to the transport was invalid.");
	case 0xc0000141u:
		return __tsc(u8"The address handle given to the transport was invalid.");
	case 0xc0000142u:
		return __tsc(u8"{DLL Initialization Failed}\nInitialization of the dynamic link library %hs failed. The process is terminating abnormally.");
	case 0xc0000143u:
		return __tsc(u8"{Missing System File}\nThe required system file %hs is bad or missing.");
	case 0xc0000144u:
		return __tsc(u8"{Application Error}\nThe exception %s (0x");
	case 0xc0000145u:
		return __tsc(u8"{Application Error}\nThe application was unable to start correctly (0x%lx). Click OK to close the application.");
	case 0xc0000146u:
		return __tsc(u8"{Unable to Create Paging File}\nThe creation of the paging file %hs failed (%lx). The requested size was %ld.");
	case 0xc0000147u:
		return __tsc(u8"{No Paging File Specified}\nNo paging file was specified in the system configuration.");
	case 0xc0000148u:
		return __tsc(u8"{Incorrect System Call Level}\nAn invalid level was passed into the specified system call.");
	case 0xc0000149u:
		return __tsc(u8"{Incorrect Password to LAN Manager Server}\nYou specified an incorrect password to a LAN Manager 2.x or MS-NET server.");
	case 0xc000014au:
		return __tsc(u8"{EXCEPTION}\nA real-mode application issued a floating-point instruction and floating-point hardware is not present.");
	case 0xc000014bu:
		return __tsc(u8"The pipe operation has failed because the other end of the pipe has been closed.");
	case 0xc000014cu:
		return __tsc(u8"{The Registry Is Corrupt}\nThe structure of one of the files that contains Registry data is corrupt, or the image of the file in memory is corrupt, or the file could not be recovered because the alternate copy or log was absent or corrupt.");
	case 0xc000014du:
		return __tsc(u8"An I/O operation initiated by the Registry failed unrecoverably. The Registry could not read in, or write out, or flush, one of the files that contain the system's image of the Registry.");
	case 0xc000014eu:
		return __tsc(u8"An event pair synchronization operation was performed using the thread specific client/server event pair object, but no event pair object was associated with the thread.");
	case 0xc000014fu:
		return __tsc(u8"The volume does not contain a recognized file system. Please make sure that all required file system drivers are loaded and that the volume is not corrupt.");
	case 0xc0000150u:
		return __tsc(u8"No serial device was successfully initialized. The serial driver will unload.");
	case 0xc0000151u:
		return __tsc(u8"The specified local group does not exist.");
	case 0xc0000152u:
		return __tsc(u8"The specified account name is not a member of the group.");
	case 0xc0000153u:
		return __tsc(u8"The specified account name is already a member of the group.");
	case 0xc0000154u:
		return __tsc(u8"The specified local group already exists.");
	case 0xc0000155u:
		return __tsc(u8"A requested type of logon (e.g., Interactive, Network, Service) is not granted by the target system's local security policy.\nPlease ask the system administrator to grant the necessary form of logon.");
	case 0xc0000156u:
		return __tsc(u8"The maximum number of secrets that may be stored in a single system has been exceeded. The length and number of secrets is limited to satisfy United States State Department export restrictions.");
	case 0xc0000157u:
		return __tsc(u8"The length of a secret exceeds the maximum length allowed. The length and number of secrets is limited to satisfy United States State Department export restrictions.");
	case 0xc0000158u:
		return __tsc(u8"The Local Security Authority (LSA) database contains an internal inconsistency.");
	case 0xc0000159u:
		return __tsc(u8"The requested operation cannot be performed in fullscreen mode.");
	case 0xc000015au:
		return __tsc(u8"During a logon attempt, the user's security context accumulated too many security IDs. This is a very unusual situation. Remove the user from some global or local groups to reduce the number of security ids to incorporate into the security context.");
	case 0xc000015bu:
		return __tsc(u8"A user has requested a type of logon (e.g., interactive or network) that has not been granted. An administrator has control over who may logon interactively and through the network.");
	case 0xc000015cu:
		return __tsc(u8"The system has attempted to load or restore a file into the registry, and the specified file is not in the format of a registry file.");
	case 0xc000015du:
		return __tsc(u8"An attempt was made to change a user password in the security account manager without providing the necessary Windows cross-encrypted password.");
	case 0xc000015eu:
		return __tsc(u8"A Windows Server has an incorrect configuration.");
	case 0xc000015fu:
		return __tsc(u8"An attempt was made to explicitly access the secondary copy of information via a device control to the Fault Tolerance driver and the secondary copy is not present in the system.");
	case 0xc0000160u:
		return __tsc(u8"A configuration registry node representing a driver service entry was ill-formed and did not contain required value entries.");
	case 0xc0000161u:
		return __tsc(u8"An illegal character was encountered. For a multi-byte character set this includes a lead byte without a succeeding trail byte. For the Unicode character set this includes the characters 0xFFFF and 0xFFFE.");
	case 0xc0000162u:
		return __tsc(u8"No mapping for the Unicode character exists in the target multi-byte code page.");
	case 0xc0000163u:
		return __tsc(u8"The Unicode character is not defined in the Unicode character set installed on the system.");
	case 0xc0000164u:
		return __tsc(u8"The paging file cannot be created on a floppy diskette.");
	case 0xc0000165u:
		return __tsc(u8"{Floppy Disk Error}\nWhile accessing a floppy disk, an ID address mark was not found.");
	case 0xc0000166u:
		return __tsc(u8"{Floppy Disk Error}\nWhile accessing a floppy disk, the track address from the sector ID field was found to be different than the track address maintained by the controller.");
	case 0xc0000167u:
		return __tsc(u8"{Floppy Disk Error}\nThe floppy disk controller reported an error that is not recognized by the floppy disk driver.");
	case 0xc0000168u:
		return __tsc(u8"{Floppy Disk Error}\nWhile accessing a floppy-disk, the controller returned inconsistent results via its registers.");
	case 0xc0000169u:
		return __tsc(u8"{Hard Disk Error}\nWhile accessing the hard disk, a recalibrate operation failed, even after retries.");
	case 0xc000016au:
		return __tsc(u8"{Hard Disk Error}\nWhile accessing the hard disk, a disk operation failed even after retries.");
	case 0xc000016bu:
		return __tsc(u8"{Hard Disk Error}\nWhile accessing the hard disk, a disk controller reset was needed, but even that failed.");
	case 0xc000016cu:
		return __tsc(u8"An attempt was made to open a device that was sharing an IRQ with other devices.\nAt least one other device that uses that IRQ was already opened.\nTwo concurrent opens of devices that share an IRQ and only work via interrupts is not supported for the particular bus type that the devices use.");
	case 0xc000016du:
		return __tsc(u8"{FT Orphaning}\nA disk that is part of a fault-tolerant volume can no longer be accessed.");
	case 0xc000016eu:
		return __tsc(u8"The system bios failed to connect a system interrupt to the device or bus for which the device is connected.");
	case 0xc0000172u:
		return __tsc(u8"Tape could not be partitioned.");
	case 0xc0000173u:
		return __tsc(u8"When accessing a new tape of a multivolume partition, the current blocksize is incorrect.");
	case 0xc0000174u:
		return __tsc(u8"Tape partition information could not be found when loading a tape.");
	case 0xc0000175u:
		return __tsc(u8"Attempt to lock the eject media mechanism fails.");
	case 0xc0000176u:
		return __tsc(u8"Unload media fails.");
	case 0xc0000177u:
		return __tsc(u8"Physical end of tape was detected.");
	case 0xc0000178u:
		return __tsc(u8"{No Media}\nThere is no media in the drive. Please insert media into drive %hs.");
	case 0xc000017au:
		return __tsc(u8"A member could not be added to or removed from the local group because the member does not exist.");
	case 0xc000017bu:
		return __tsc(u8"A new member could not be added to a local group because the member has the wrong account type.");
	case 0xc000017cu:
		return __tsc(u8"Illegal operation attempted on a registry key which has been marked for deletion.");
	case 0xc000017du:
		return __tsc(u8"System could not allocate required space in a registry log.");
	case 0xc000017eu:
		return __tsc(u8"Too many Sids have been specified.");
	case 0xc000017fu:
		return __tsc(u8"An attempt was made to change a user password in the security account manager without providing the necessary LM cross-encrypted password.");
	case 0xc0000180u:
		return __tsc(u8"An attempt was made to create a symbolic link in a registry key that already has subkeys or values.");
	case 0xc0000181u:
		return __tsc(u8"An attempt was made to create a Stable subkey under a Volatile parent key.");
	case 0xc0000182u:
		return __tsc(u8"The I/O device is configured incorrectly or the configuration parameters to the driver are incorrect.");
	case 0xc0000183u:
		return __tsc(u8"An error was detected between two drivers or within an I/O driver.");
	case 0xc0000184u:
		return __tsc(u8"The device is not in a valid state to perform this request.");
	case 0xc0000185u:
		return __tsc(u8"The I/O device reported an I/O error.");
	case 0xc0000186u:
		return __tsc(u8"A protocol error was detected between the driver and the device.");
	case 0xc0000187u:
		return __tsc(u8"This operation is only allowed for the Primary Domain Controller of the domain.");
	case 0xc0000188u:
		return __tsc(u8"Log file space is insufficient to support this operation.");
	case 0xc0000189u:
		return __tsc(u8"A write operation was attempted to a volume after it was dismounted.");
	case 0xc000018au:
		return __tsc(u8"The workstation does not have a trust secret for the primary domain in the local LSA database.");
	case 0xc000018bu:
		return __tsc(u8"The SAM database on the Windows Server does not have a computer account for this workstation trust relationship.");
	case 0xc000018cu:
		return __tsc(u8"The logon request failed because the trust relationship between the primary domain and the trusted domain failed.");
	case 0xc000018du:
		return __tsc(u8"The logon request failed because the trust relationship between this workstation and the primary domain failed.");
	case 0xc000018eu:
		return __tsc(u8"The Eventlog log file is corrupt.");
	case 0xc000018fu:
		return __tsc(u8"No Eventlog log file could be opened. The Eventlog service did not start.");
	case 0xc0000190u:
		return __tsc(u8"The network logon failed. This may be because the validation authority can't be reached.");
	case 0xc0000191u:
		return __tsc(u8"An attempt was made to acquire a mutant such that its maximum count would have been exceeded.");
	case 0xc0000192u:
		return __tsc(u8"An attempt was made to logon, but the netlogon service was not started.");
	case 0xc0000193u:
		return __tsc(u8"The user's account has expired.");
	case 0xc0000194u:
		return __tsc(u8"{EXCEPTION}\nPossible deadlock condition.");
	case 0xc0000195u:
		return __tsc(u8"Multiple connections to a server or shared resource by the same user, using more than one user name, are not allowed. Disconnect all previous connections to the server or shared resource and try again.");
	case 0xc0000196u:
		return __tsc(u8"An attempt was made to establish a session to a network server, but there are already too many sessions established to that server.");
	case 0xc0000197u:
		return __tsc(u8"The log file has changed between reads.");
	case 0xc0000198u:
		return __tsc(u8"The account used is an Interdomain Trust account. Use your global user account or local user account to access this server.");
	case 0xc0000199u:
		return __tsc(u8"The account used is a Computer Account. Use your global user account or local user account to access this server.");
	case 0xc000019au:
		return __tsc(u8"The account used is an Server Trust account. Use your global user account or local user account to access this server.");
	case 0xc000019bu:
		return __tsc(u8"The name or SID of the domain specified is inconsistent with the trust information for that domain.");
	case 0xc000019cu:
		return __tsc(u8"A volume has been accessed for which a file system driver is required that has not yet been loaded.");
	case 0xc000019du:
		return __tsc(u8"Indicates that the specified image is already loaded as a DLL.");
	case 0xc000019eu:
		return __tsc(u8"Short name settings may not be changed on this volume due to the global registry setting.");
	case 0xc000019fu:
		return __tsc(u8"Short names are not enabled on this volume.");
	case 0xc00001a0u:
		return __tsc(u8"The security stream for the given volume is in an inconsistent state.\nPlease run CHKDSK on the volume.");
	case 0xc00001a1u:
		return __tsc(u8"A requested file lock operation cannot be processed due to an invalid byte range.");
	case 0xc00001a2u:
		return __tsc(u8"{Invalid ACE Condition}\nThe specified access control entry (ACE) contains an invalid condition.");
	case 0xc00001a3u:
		return __tsc(u8"The subsystem needed to support the image type is not present.");
	case 0xc00001a4u:
		return __tsc(u8"{Invalid ACE Condition}\nThe specified file already has a notification GUID associated with it.");
	case 0xc00001a5u:
		return __tsc(u8"An invalid exception handler routine has been detected.");
	case 0xc00001a6u:
		return __tsc(u8"Duplicate privileges were specified for the token.");
	case 0xc00001a7u:
		return __tsc(u8"Requested action not allowed on a file system internal file.");
	case 0xc00001a8u:
		return __tsc(u8"A portion of the file system requires repair.");
	case 0xc00001a9u:
		return __tsc(u8"Quota support is not enabled on the system.");
	case 0xc00001aau:
		return __tsc(u8"The operation failed because the application is not part of an application package.");
	case 0xc00001abu:
		return __tsc(u8"File metadata optimization is already in progress.");
	case 0xc00001acu:
		return __tsc(u8"The objects are not identical.");
	case 0xc00001adu:
		return __tsc(u8"The process has terminated because it could not allocate additional memory.");
	case 0xc00001aeu:
		return __tsc(u8"The process is not part of a job.");
	case 0xc00001afu:
		return __tsc(u8"The specified CPU Set IDs are invalid.");
	case 0xc00001b0u:
	case 0xc00001b1u:
	case 0xc00001b2u:
	case 0xc00001b3u:
	case 0xc00001b4u:
	case 0xc00001b5u:
		return __tsc(u8"");
	case 0xc0000201u:
		return __tsc(u8"A remote open failed because the network open restrictions were not satisfied.");
	case 0xc0000202u:
		return __tsc(u8"There is no user session key for the specified logon session.");
	case 0xc0000203u:
		return __tsc(u8"The remote user session has been deleted.");
	case 0xc0000204u:
		return __tsc(u8"Indicates the specified resource language ID cannot be found in the\nimage file.");
	case 0xc0000205u:
		return __tsc(u8"Insufficient server resources exist to complete the request.");
	case 0xc0000206u:
		return __tsc(u8"The size of the buffer is invalid for the specified operation.");
	case 0xc0000207u:
		return __tsc(u8"The transport rejected the network address specified as invalid.");
	case 0xc0000208u:
		return __tsc(u8"The transport rejected the network address specified due to an invalid use of a wildcard.");
	case 0xc0000209u:
		return __tsc(u8"The transport address could not be opened because all the available addresses are in use.");
	case 0xc000020au:
		return __tsc(u8"The transport address could not be opened because it already exists.");
	case 0xc000020bu:
		return __tsc(u8"The transport address is now closed.");
	case 0xc000020cu:
		return __tsc(u8"The transport connection is now disconnected.");
	case 0xc000020du:
		return __tsc(u8"The transport connection has been reset.");
	case 0xc000020eu:
		return __tsc(u8"The transport cannot dynamically acquire any more nodes.");
	case 0xc000020fu:
		return __tsc(u8"The transport aborted a pending transaction.");
	case 0xc0000210u:
		return __tsc(u8"The transport timed out a request waiting for a response.");
	case 0xc0000211u:
		return __tsc(u8"The transport did not receive a release for a pending response.");
	case 0xc0000212u:
		return __tsc(u8"The transport did not find a transaction matching the specific token.");
	case 0xc0000213u:
		return __tsc(u8"The transport had previously responded to a transaction request.");
	case 0xc0000214u:
		return __tsc(u8"The transport does not recognized the transaction request identifier specified.");
	case 0xc0000215u:
		return __tsc(u8"The transport does not recognize the transaction request type specified.");
	case 0xc0000216u:
		return __tsc(u8"The transport can only process the specified request on the server side of a session.");
	case 0xc0000217u:
		return __tsc(u8"The transport can only process the specified request on the client side of a session.");
	case 0xc0000218u:
		return __tsc(u8"{Registry File Failure}\nThe registry cannot load the hive (file):\n%hs\nor its log or alternate.\nIt is corrupt, absent, or not writable.");
	case 0xc0000219u:
		return __tsc(u8"{Unexpected Failure in DebugActiveProcess}\nAn unexpected failure occurred while processing a DebugActiveProcess API request. You may choose OK to terminate the process, or Cancel to ignore the error.");
	case 0xc000021au:
		return __tsc(u8"{Fatal System Error}\nThe %hs system process terminated unexpectedly with a status of 0x");
	case 0xc000021bu:
		return __tsc(u8"{Data Not Accepted}\nThe TDI client could not handle the data received during an indication.");
	case 0xc000021cu:
		return __tsc(u8"{Unable to Retrieve Browser Server List}\nThe list of servers for this workgroup is not currently available.");
	case 0xc000021du:
		return __tsc(u8"NTVDM encountered a hard error.");
	case 0xc000021eu:
		return __tsc(u8"{Cancel Timeout}\nThe driver %hs failed to complete a cancelled I/O request in the allotted time.");
	case 0xc000021fu:
		return __tsc(u8"{Reply Message Mismatch}\nAn attempt was made to reply to an LPC message, but the thread specified by the client ID in the message was not waiting on that message.");
	case 0xc0000220u:
		return __tsc(u8"{Mapped View Alignment Incorrect}\nAn attempt was made to map a view of a file, but either the specified base address or the offset into the file were not aligned on the proper allocation granularity.");
	case 0xc0000221u:
		return __tsc(u8"{Bad Image Checksum}\nThe image %hs is possibly corrupt. The header checksum does not match the computed checksum.");
	case 0xc0000222u:
		return __tsc(u8"{Delayed Write Failed}\nWindows was unable to save all the data for the file %hs. The data has been lost. This error may be caused by a failure of your computer hardware or network connection. Please try to save this file elsewhere.");
	case 0xc0000223u:
		return __tsc(u8"The parameter(s) passed to the server in the client/server shared memory window were invalid. Too much data may have been put in the shared memory window.");
	case 0xc0000224u:
		return __tsc(u8"The user's password must be changed before signing in.");
	case 0xc0000225u:
		return __tsc(u8"The object was not found.");
	case 0xc0000226u:
		return __tsc(u8"The stream is not a tiny stream.");
	case 0xc0000227u:
		return __tsc(u8"A transaction recover failed.");
	case 0xc0000228u:
		return __tsc(u8"The request must be handled by the stack overflow code.");
	case 0xc0000229u:
		return __tsc(u8"A consistency check failed.");
	case 0xc000022au:
		return __tsc(u8"The attempt to insert the ID in the index failed because the ID is already in the index.");
	case 0xc000022bu:
		return __tsc(u8"The attempt to set the object's ID failed because the object already has an ID.");
	case 0xc000022cu:
		return __tsc(u8"Internal OFS status codes indicating how an allocation operation is handled. Either it is retried after the containing onode is moved or the extent stream is converted to a large stream.");
	case 0xc000022du:
		return __tsc(u8"The request needs to be retried.");
	case 0xc000022eu:
		return __tsc(u8"The attempt to find the object found an object matching by ID on the volume but it is out of the scope of the handle used for the operation.");
	case 0xc000022fu:
		return __tsc(u8"The bucket array must be grown. Retry transaction after doing so.");
	case 0xc0000230u:
		return __tsc(u8"The property set specified does not exist on the object.");
	case 0xc0000231u:
		return __tsc(u8"The user/kernel marshalling buffer has overflowed.");
	case 0xc0000232u:
		return __tsc(u8"The supplied variant structure contains invalid data.");
	case 0xc0000233u:
		return __tsc(u8"Could not find a domain controller for this domain.");
	case 0xc0000234u:
		return __tsc(u8"The user account has been automatically locked because too many invalid logon attempts or password change attempts have been requested.");
	case 0xc0000235u:
		return __tsc(u8"NtClose was called on a handle that was protected from close via NtSetInformationObject.");
	case 0xc0000236u:
		return __tsc(u8"The transport connection attempt was refused by the remote system.");
	case 0xc0000237u:
		return __tsc(u8"The transport connection was gracefully closed.");
	case 0xc0000238u:
		return __tsc(u8"The transport endpoint already has an address associated with it.");
	case 0xc0000239u:
		return __tsc(u8"An address has not yet been associated with the transport endpoint.");
	case 0xc000023au:
		return __tsc(u8"An operation was attempted on a nonexistent transport connection.");
	case 0xc000023bu:
		return __tsc(u8"An invalid operation was attempted on an active transport connection.");
	case 0xc000023cu:
		return __tsc(u8"The remote network is not reachable by the transport.");
	case 0xc000023du:
		return __tsc(u8"The remote system is not reachable by the transport.");
	case 0xc000023eu:
		return __tsc(u8"The remote system does not support the transport protocol.");
	case 0xc000023fu:
		return __tsc(u8"No service is operating at the destination port of the transport on the remote system.");
	case 0xc0000240u:
		return __tsc(u8"The request was aborted.");
	case 0xc0000241u:
		return __tsc(u8"The transport connection was aborted by the local system.");
	case 0xc0000242u:
		return __tsc(u8"The specified buffer contains ill-formed data.");
	case 0xc0000243u:
		return __tsc(u8"The requested operation cannot be performed on a file with a user mapped section open.");
	case 0xc0000244u:
		return __tsc(u8"{Audit Failed}\nAn attempt to generate a security audit failed.");
	case 0xc0000245u:
		return __tsc(u8"The timer resolution was not previously set by the current process.");
	case 0xc0000246u:
		return __tsc(u8"A connection to the server could not be made because the limit on the number of concurrent connections for this account has been reached.");
	case 0xc0000247u:
		return __tsc(u8"Attempting to login during an unauthorized time of day for this account.");
	case 0xc0000248u:
		return __tsc(u8"The account is not authorized to login from this station.");
	case 0xc0000249u:
		return __tsc(u8"{UP/MP Image Mismatch}\nThe image %hs has been modified for use on a uniprocessor system, but you are running it on a multiprocessor machine.\nPlease reinstall the image file.");
	case 0xc0000250u:
		return __tsc(u8"There is insufficient account information to log you on.");
	case 0xc0000251u:
		return __tsc(u8"{Invalid DLL Entrypoint}\nThe dynamic link library %hs is not written correctly. The stack pointer has been left in an inconsistent state. The entrypoint should be declared as WINAPI or STDCALL. Select YES to fail the DLL load. Select NO to continue execution. Selecting NO may cause the application to operate incorrectly.");
	case 0xc0000252u:
		return __tsc(u8"{Invalid Service Callback Entrypoint}\nThe %hs service is not written correctly. The stack pointer has been left in an inconsistent state. The callback entrypoint should be declared as WINAPI or STDCALL. Selecting OK will cause the service to continue operation. However, the service process may operate incorrectly.");
	case 0xc0000253u:
		return __tsc(u8"The server received the messages but did not send a reply.");
	case 0xc0000254u:
	case 0xc0000255u:
		return __tsc(u8"There is an IP address conflict with another system on the network");
	case 0xc0000256u:
		return __tsc(u8"{Low On Registry Space}\nThe system has reached the maximum size allowed for the system part of the registry. Additional storage requests will be ignored.");
	case 0xc0000257u:
		return __tsc(u8"The contacted server does not support the indicated part of the DFS namespace.");
	case 0xc0000258u:
		return __tsc(u8"A callback return system service cannot be executed when no callback is active.");
	case 0xc0000259u:
		return __tsc(u8"The service being accessed is licensed for a particular number of connections. No more connections can be made to the service at this time because there are already as many connections as the service can accept.");
	case 0xc000025au:
		return __tsc(u8"The password provided is too short to meet the policy of your user account. Please choose a longer password.");
	case 0xc000025bu:
		return __tsc(u8"The policy of your user account does not allow you to change passwords too frequently. This is done to prevent users from changing back to a familiar, but potentially discovered, password. If you feel your password has been compromised then please contact your administrator immediately to have a new one assigned.");
	case 0xc000025cu:
		return __tsc(u8"You have attempted to change your password to one that you have used in the past. The policy of your user account does not allow this. Please select a password that you have not previously used.");
	case 0xc000025eu:
		return __tsc(u8"You have attempted to load a legacy device driver while its device instance had been disabled.");
	case 0xc000025fu:
		return __tsc(u8"The specified compression format is unsupported.");
	case 0xc0000260u:
		return __tsc(u8"The specified hardware profile configuration is invalid.");
	case 0xc0000261u:
		return __tsc(u8"The specified Plug and Play registry device path is invalid.");
	case 0xc0000262u:
		return __tsc(u8"{Driver Entry Point Not Found}\nThe %hs device driver could not locate the ordinal %ld in driver %hs.");
	case 0xc0000263u:
		return __tsc(u8"{Driver Entry Point Not Found}\nThe %hs device driver could not locate the entry point %hs in driver %hs.");
	case 0xc0000264u:
		return __tsc(u8"{Application Error}\nThe application attempted to release a resource it did not own. Click OK to terminate the application.");
	case 0xc0000265u:
		return __tsc(u8"An attempt was made to create more links on a file than the file system supports.");
	case 0xc0000266u:
		return __tsc(u8"The specified quota list is internally inconsistent with its descriptor.");
	case 0xc0000267u:
		return __tsc(u8"The specified file has been relocated to offline storage.");
	case 0xc0000268u:
		return __tsc(u8"{Windows Evaluation Notification}\nThe evaluation period for this installation of Windows has expired. This system will shutdown in 1 hour. To restore access to this installation of Windows, please upgrade this installation using a licensed distribution of this product.");
	case 0xc0000269u:
		return __tsc(u8"{Illegal System DLL Relocation}\nThe system DLL %hs was relocated in memory. The application will not run properly. The relocation occurred because the DLL %hs occupied an address range reserved for Windows system DLLs. The vendor supplying the DLL should be contacted for a new DLL.");
	case 0xc000026au:
		return __tsc(u8"{License Violation}\nThe system has detected tampering with your registered product type. This is a violation of your software license. Tampering with product type is not permitted.");
	case 0xc000026bu:
		return __tsc(u8"{DLL Initialization Failed}\nThe application failed to initialize because the window station is shutting down.");
	case 0xc000026cu:
		return __tsc(u8"{Unable to Load Device Driver}\n%hs device driver could not be loaded.\nError Status was 0x%x");
	case 0xc000026du:
		return __tsc(u8"DFS is unavailable on the contacted server.");
	case 0xc000026eu:
		return __tsc(u8"An operation was attempted to a volume after it was dismounted.");
	case 0xc000026fu:
		return __tsc(u8"An internal error occurred in the Win32 x86 emulation subsystem.");
	case 0xc0000270u:
		return __tsc(u8"Win32 x86 emulation subsystem Floating-point stack check.");
	case 0xc0000271u:
		return __tsc(u8"The validation process needs to continue on to the next step.");
	case 0xc0000272u:
		return __tsc(u8"There was no match for the specified key in the index.");
	case 0xc0000273u:
		return __tsc(u8"There are no more matches for the current index enumeration.");
	case 0xc0000275u:
		return __tsc(u8"The NTFS file or directory is not a reparse point.");
	case 0xc0000276u:
		return __tsc(u8"The Windows I/O reparse tag passed for the NTFS reparse point is invalid.");
	case 0xc0000277u:
		return __tsc(u8"The Windows I/O reparse tag does not match the one present in the NTFS reparse point.");
	case 0xc0000278u:
		return __tsc(u8"The user data passed for the NTFS reparse point is invalid.");
	case 0xc0000279u:
		return __tsc(u8"The layered file system driver for this IO tag did not handle it when needed.");
	case 0xc000027au:
		return __tsc(u8"The password provided is too long to meet the policy of your user account. Please choose a shorter password.");
	case 0xc000027bu:
		return __tsc(u8"An application-internal exception has occurred.");
	case 0xc000027cu:
		return __tsc(u8"");
	case 0xc0000280u:
		return __tsc(u8"The NTFS symbolic link could not be resolved even though the initial file name is valid.");
	case 0xc0000281u:
		return __tsc(u8"The NTFS directory is a reparse point.");
	case 0xc0000282u:
		return __tsc(u8"The range could not be added to the range list because of a conflict.");
	case 0xc0000283u:
		return __tsc(u8"The specified medium changer source element contains no media.");
	case 0xc0000284u:
		return __tsc(u8"The specified medium changer destination element already contains media.");
	case 0xc0000285u:
		return __tsc(u8"The specified medium changer element does not exist.");
	case 0xc0000286u:
		return __tsc(u8"The specified element is contained within a magazine that is no longer present.");
	case 0xc0000287u:
		return __tsc(u8"The device requires reinitialization due to hardware errors.");
	case 0xc000028au:
		return __tsc(u8"The file encryption attempt failed.");
	case 0xc000028bu:
		return __tsc(u8"The file decryption attempt failed.");
	case 0xc000028cu:
		return __tsc(u8"The specified range could not be found in the range list.");
	case 0xc000028du:
		return __tsc(u8"There is no encryption recovery policy configured for this system.");
	case 0xc000028eu:
		return __tsc(u8"The required encryption driver is not loaded for this system.");
	case 0xc000028fu:
		return __tsc(u8"The file was encrypted with a different encryption driver than is currently loaded.");
	case 0xc0000290u:
		return __tsc(u8"There are no EFS keys defined for the user.");
	case 0xc0000291u:
		return __tsc(u8"The specified file is not encrypted.");
	case 0xc0000292u:
		return __tsc(u8"The specified file is not in the defined EFS export format.");
	case 0xc0000293u:
		return __tsc(u8"The specified file is encrypted and the user does not have the ability to decrypt it.");
	case 0xc0000295u:
		return __tsc(u8"The guid passed was not recognized as valid by a WMI data provider.");
	case 0xc0000296u:
		return __tsc(u8"The instance name passed was not recognized as valid by a WMI data provider.");
	case 0xc0000297u:
		return __tsc(u8"The data item id passed was not recognized as valid by a WMI data provider.");
	case 0xc0000298u:
		return __tsc(u8"The WMI request could not be completed and should be retried.");
	case 0xc0000299u:
		return __tsc(u8"The policy object is shared and can only be modified at the root");
	case 0xc000029au:
		return __tsc(u8"The policy object does not exist when it should");
	case 0xc000029bu:
		return __tsc(u8"The requested policy information only lives in the Ds");
	case 0xc000029cu:
		return __tsc(u8"The volume must be upgraded to enable this feature");
	case 0xc000029du:
		return __tsc(u8"The remote storage service is not operational at this time.");
	case 0xc000029eu:
		return __tsc(u8"The remote storage service encountered a media error.");
	case 0xc000029fu:
		return __tsc(u8"The tracking (workstation) service is not running.");
	case 0xc00002a0u:
		return __tsc(u8"The server process is running under a SID different than that required by client.");
	case 0xc00002a1u:
		return __tsc(u8"The specified directory service attribute or value does not exist.");
	case 0xc00002a2u:
		return __tsc(u8"The attribute syntax specified to the directory service is invalid.");
	case 0xc00002a3u:
		return __tsc(u8"The attribute type specified to the directory service is not defined.");
	case 0xc00002a4u:
		return __tsc(u8"The specified directory service attribute or value already exists.");
	case 0xc00002a5u:
		return __tsc(u8"The directory service is busy.");
	case 0xc00002a6u:
		return __tsc(u8"The directory service is not available.");
	case 0xc00002a7u:
		return __tsc(u8"The directory service was unable to allocate a relative identifier.");
	case 0xc00002a8u:
		return __tsc(u8"The directory service has exhausted the pool of relative identifiers.");
	case 0xc00002a9u:
		return __tsc(u8"The requested operation could not be performed because the directory service is not the master for that type of operation.");
	case 0xc00002aau:
		return __tsc(u8"The directory service was unable to initialize the subsystem that allocates relative identifiers.");
	case 0xc00002abu:
		return __tsc(u8"The requested operation did not satisfy one or more constraints associated with the class of the object.");
	case 0xc00002acu:
		return __tsc(u8"The directory service can perform the requested operation only on a leaf object.");
	case 0xc00002adu:
		return __tsc(u8"The directory service cannot perform the requested operation on the Relatively Defined Name (RDN) attribute of an object.");
	case 0xc00002aeu:
		return __tsc(u8"The directory service detected an attempt to modify the object class of an object.");
	case 0xc00002afu:
		return __tsc(u8"An error occurred while performing a cross domain move operation.");
	case 0xc00002b0u:
		return __tsc(u8"Unable to Contact the Global Catalog Server.");
	case 0xc00002b1u:
		return __tsc(u8"The requested operation requires a directory service, and none was available.");
	case 0xc00002b2u:
		return __tsc(u8"The reparse attribute cannot be set as it is incompatible with an existing attribute.");
	case 0xc00002b3u:
		return __tsc(u8"A group marked use for deny only cannot be enabled.");
	case 0xc00002b4u:
		return __tsc(u8"{EXCEPTION}\nMultiple floating point faults.");
	case 0xc00002b5u:
		return __tsc(u8"{EXCEPTION}\nMultiple floating point traps.");
	case 0xc00002b6u:
		return __tsc(u8"The device has been removed.");
	case 0xc00002b7u:
		return __tsc(u8"The volume change journal is being deleted.");
	case 0xc00002b8u:
		return __tsc(u8"The volume change journal is not active.");
	case 0xc00002b9u:
		return __tsc(u8"The requested interface is not supported.");
	case 0xc00002bau:
		return __tsc(u8"The directory service detected the subsystem that allocates relative identifiers is disabled. This can occur as a protective mechanism when the system determines a significant portion of relative identifiers (RIDs) have been exhausted. Please see http://go.microsoft.com/fwlink/?LinkId=228610 for recommended diagnostic steps and the procedure to re-enable account creation.");
	case 0xc00002c1u:
		return __tsc(u8"A directory service resource limit has been exceeded.");
	case 0xc00002c2u:
		return __tsc(u8"{System Standby Failed}\nThe driver %hs does not support standby mode. Updating this driver may allow the system to go to standby mode.");
	case 0xc00002c3u:
		return __tsc(u8"Mutual Authentication failed. The server's password is out of date at the domain controller.");
	case 0xc00002c4u:
		return __tsc(u8"The system file %1 has become corrupt and has been replaced.");
	case 0xc00002c5u:
		return __tsc(u8"{EXCEPTION}\nAlignment Error\nA datatype misalignment error was detected in a load or store instruction.");
	case 0xc00002c6u:
		return __tsc(u8"The WMI data item or data block is read only.");
	case 0xc00002c7u:
		return __tsc(u8"The WMI data item or data block could not be changed.");
	case 0xc00002c8u:
		return __tsc(u8"{Virtual Memory Minimum Too Low}\nYour system is low on virtual memory. Windows is increasing the size of your virtual memory paging file. During this process, memory requests for some applications may be denied. For more information, see Help.");
	case 0xc00002c9u:
		return __tsc(u8"{EXCEPTION}\nRegister NaT consumption faults.\nA NaT value is consumed on a non speculative instruction.");
	case 0xc00002cau:
		return __tsc(u8"The medium changer's transport element contains media, which is causing the operation to fail.");
	case 0xc00002cbu:
		return __tsc(u8"Security Accounts Manager initialization failed because of the following error:\n%hs\nError Status: 0x%x.\nPlease shutdown this system and reboot into Directory Services Restore Mode, check the event log for more detailed information.");
	case 0xc00002ccu:
		return __tsc(u8"This operation is supported only when you are connected to the server.");
	case 0xc00002cdu:
		return __tsc(u8"Only an administrator can modify the membership list of an administrative group.");
	case 0xc00002ceu:
		return __tsc(u8"A device was removed so enumeration must be restarted.");
	case 0xc00002cfu:
		return __tsc(u8"The journal entry has been deleted from the journal.");
	case 0xc00002d0u:
		return __tsc(u8"Cannot change the primary group ID of a domain controller account.");
	case 0xc00002d1u:
		return __tsc(u8"{Fatal System Error}\nThe system image %s is not properly signed. The file has been replaced with the signed file. The system has been shut down.");
	case 0xc00002d2u:
		return __tsc(u8"Device will not start without a reboot.");
	case 0xc00002d3u:
		return __tsc(u8"Current device power state cannot support this request.");
	case 0xc00002d4u:
		return __tsc(u8"The specified group type is invalid.");
	case 0xc00002d5u:
		return __tsc(u8"In mixed domain no nesting of global group if group is security enabled.");
	case 0xc00002d6u:
		return __tsc(u8"In mixed domain, cannot nest local groups with other local groups, if the group is security enabled.");
	case 0xc00002d7u:
		return __tsc(u8"A global group cannot have a local group as a member.");
	case 0xc00002d8u:
		return __tsc(u8"A global group cannot have a universal group as a member.");
	case 0xc00002d9u:
		return __tsc(u8"A universal group cannot have a local group as a member.");
	case 0xc00002dau:
		return __tsc(u8"A global group cannot have a cross domain member.");
	case 0xc00002dbu:
		return __tsc(u8"A local group cannot have another cross domain local group as a member.");
	case 0xc00002dcu:
		return __tsc(u8"Cannot change to security disabled group because of having primary members in this group.");
	case 0xc00002ddu:
		return __tsc(u8"The WMI operation is not supported by the data block or method.");
	case 0xc00002deu:
		return __tsc(u8"There is not enough power to complete the requested operation.");
	case 0xc00002dfu:
		return __tsc(u8"Security Account Manager needs to get the boot password.");
	case 0xc00002e0u:
		return __tsc(u8"Security Account Manager needs to get the boot key from floppy disk.");
	case 0xc00002e1u:
		return __tsc(u8"Directory Service cannot start.");
	case 0xc00002e2u:
		return __tsc(u8"Directory Services could not start because of the following error:\n%hs\nError Status: 0x%x.\nPlease shutdown this system and reboot into Directory Services Restore Mode, check the event log for more detailed information.");
	case 0xc00002e3u:
		return __tsc(u8"Security Accounts Manager initialization failed because of the following error:\n%hs\nError Status: 0x%x.\nPlease click OK to shutdown this system and reboot into Safe Mode, check the event log for more detailed information.");
	case 0xc00002e4u:
		return __tsc(u8"The requested operation can be performed only on a global catalog server.");
	case 0xc00002e5u:
		return __tsc(u8"A local group can only be a member of other local groups in the same domain.");
	case 0xc00002e6u:
		return __tsc(u8"Foreign security principals cannot be members of universal groups.");
	case 0xc00002e7u:
		return __tsc(u8"Your computer could not be joined to the domain. You have exceeded the maximum number of computer accounts you are allowed to create in this domain. Contact your system administrator to have this limit reset or increased.");
	case 0xc00002e8u:
		return __tsc(u8"STATUS_MULTIPLE_FAULT_VIOLATION");
	case 0xc00002e9u:
		return __tsc(u8"This operation cannot be performed on the current domain.");
	case 0xc00002eau:
		return __tsc(u8"The directory or file cannot be created.");
	case 0xc00002ebu:
		return __tsc(u8"The system is in the process of shutting down.");
	case 0xc00002ecu:
		return __tsc(u8"Directory Services could not start because of the following error:\n%hs\nError Status: 0x%x.\nPlease click OK to shutdown the system. You can use the recovery console to diagnose the system further.");
	case 0xc00002edu:
		return __tsc(u8"Security Accounts Manager initialization failed because of the following error:\n%hs\nError Status: 0x%x.\nPlease click OK to shutdown the system. You can use the recovery console to diagnose the system further.");
	case 0xc00002eeu:
		return __tsc(u8"A security context was deleted before the context was completed. This is considered a logon failure.");
	case 0xc00002efu:
		return __tsc(u8"The client is trying to negotiate a context and the server requires user-to-user but didn't send a TGT reply.");
	case 0xc00002f0u:
		return __tsc(u8"An object ID was not found in the file.");
	case 0xc00002f1u:
		return __tsc(u8"Unable to accomplish the requested task because the local machine does not have any IP addresses.");
	case 0xc00002f2u:
		return __tsc(u8"The supplied credential handle does not match the credential associated with the security context.");
	case 0xc00002f3u:
		return __tsc(u8"The crypto system or checksum function is invalid because a required function is unavailable.");
	case 0xc00002f4u:
		return __tsc(u8"The number of maximum ticket referrals has been exceeded.");
	case 0xc00002f5u:
		return __tsc(u8"The local machine must be a Kerberos KDC (domain controller) and it is not.");
	case 0xc00002f6u:
		return __tsc(u8"The other end of the security negotiation is requires strong crypto but it is not supported on the local machine.");
	case 0xc00002f7u:
		return __tsc(u8"The KDC reply contained more than one principal name.");
	case 0xc00002f8u:
		return __tsc(u8"Expected to find PA data for a hint of what etype to use, but it was not found.");
	case 0xc00002f9u:
		return __tsc(u8"The client certificate does not contain a valid UPN, or does not match the client name in the logon request. Please contact your administrator.");
	case 0xc00002fau:
		return __tsc(u8"Smartcard logon is required and was not used.");
	case 0xc00002fbu:
		return __tsc(u8"An invalid request was sent to the KDC.");
	case 0xc00002fcu:
		return __tsc(u8"The KDC was unable to generate a referral for the service requested.");
	case 0xc00002fdu:
		return __tsc(u8"The encryption type requested is not supported by the KDC.");
	case 0xc00002feu:
		return __tsc(u8"A system shutdown is in progress.");
	case 0xc00002ffu:
		return __tsc(u8"The server machine is shutting down.");
	case 0xc0000300u:
		return __tsc(u8"This operation is not supported on a computer running Windows Server 2003 for Small Business Server");
	case 0xc0000301u:
		return __tsc(u8"The WMI GUID is no longer available");
	case 0xc0000302u:
		return __tsc(u8"Collection or events for the WMI GUID is already disabled.");
	case 0xc0000303u:
		return __tsc(u8"Collection or events for the WMI GUID is already enabled.");
	case 0xc0000304u:
		return __tsc(u8"The Master File Table on the volume is too fragmented to complete this operation.");
	case 0xc0000305u:
		return __tsc(u8"Copy protection failure.");
	case 0xc0000306u:
		return __tsc(u8"Copy protection error - DVD CSS Authentication failed.");
	case 0xc0000307u:
		return __tsc(u8"Copy protection error - The given sector does not contain a valid key.");
	case 0xc0000308u:
		return __tsc(u8"Copy protection error - DVD session key not established.");
	case 0xc0000309u:
		return __tsc(u8"Copy protection error - The read failed because the sector is encrypted.");
	case 0xc000030au:
		return __tsc(u8"Copy protection error - The given DVD's region does not correspond to the\nregion setting of the drive.");
	case 0xc000030bu:
		return __tsc(u8"Copy protection error - The drive's region setting may be permanent.");
	case 0xc000030cu:
		return __tsc(u8"EAS policy requires that the user change their password before this operation can be performed.");
	case 0xc000030du:
		return __tsc(u8"An administrator has restricted sign in. To sign in, make sure your device is connected to the Internet, and have your administrator sign in first.");
	case 0xc0000320u:
		return __tsc(u8"The Kerberos protocol encountered an error while validating the KDC certificate during logon. There is more information in the system event log.");
	case 0xc0000321u:
		return __tsc(u8"The Kerberos protocol encountered an error while attempting to utilize the smartcard subsystem.");
	case 0xc0000322u:
		return __tsc(u8"The target server does not have acceptable Kerberos credentials.");
	case 0xc0000350u:
		return __tsc(u8"The transport determined that the remote system is down.");
	case 0xc0000351u:
		return __tsc(u8"An unsupported preauthentication mechanism was presented to the Kerberos package.");
	case 0xc0000352u:
		return __tsc(u8"The encryption algorithm used on the source file needs a bigger key buffer than the one used on the destination file.");
	case 0xc0000353u:
		return __tsc(u8"An attempt to remove a process's DebugPort was made, but a port was not already associated with the process.");
	case 0xc0000354u:
		return __tsc(u8"Debugger Inactive: Windows may have been started without kernel debugging enabled.");
	case 0xc0000355u:
		return __tsc(u8"This version of Windows is not compatible with the behavior version of directory forest, domain or domain controller.");
	case 0xc0000356u:
		return __tsc(u8"The specified event is currently not being audited.");
	case 0xc0000357u:
		return __tsc(u8"The machine account was created pre-NT4. The account needs to be recreated.");
	case 0xc0000358u:
		return __tsc(u8"A account group cannot have a universal group as a member.");
	case 0xc0000359u:
		return __tsc(u8"The specified image file did not have the correct format, it appears to be a 32-bit Windows image.");
	case 0xc000035au:
		return __tsc(u8"The specified image file did not have the correct format, it appears to be a 64-bit Windows image.");
	case 0xc000035bu:
		return __tsc(u8"Client's supplied SSPI channel bindings were incorrect.");
	case 0xc000035cu:
		return __tsc(u8"The client's session has expired, so the client must reauthenticate to continue accessing the remote resources.");
	case 0xc000035du:
		return __tsc(u8"AppHelp dialog canceled thus preventing the application from starting.");
	case 0xc000035eu:
		return __tsc(u8"The SID filtering operation removed all SIDs.");
	case 0xc000035fu:
		return __tsc(u8"The driver was not loaded because the system is booting into safe mode.");
	case 0xc0000361u:
		return __tsc(u8"Access to %1 has been restricted by your Administrator by the default software restriction policy level.");
	case 0xc0000362u:
		return __tsc(u8"Access to %1 has been restricted by your Administrator by location with policy rule %2 placed on path %3");
	case 0xc0000363u:
		return __tsc(u8"Access to %1 has been restricted by your Administrator by software publisher policy.");
	case 0xc0000364u:
		return __tsc(u8"Access to %1 has been restricted by your Administrator by policy rule %2.");
	case 0xc0000365u:
		return __tsc(u8"The driver was not loaded because it failed its initialization call.");
	case 0xc0000366u:
		return __tsc(u8"The \"%hs\" encountered an error while applying power or reading the device configuration. This may be caused by a failure of your hardware or by a poor connection.");
	case 0xc0000368u:
		return __tsc(u8"The create operation failed because the name contained at least one mount point which resolves to a volume to which the specified device object is not attached.");
	case 0xc0000369u:
		return __tsc(u8"The device object parameter is either not a valid device object or is not attached to the volume specified by the file name.");
	case 0xc000036au:
		return __tsc(u8"A Machine Check Error has occurred. Please check the system eventlog for additional information.");
	case 0xc000036bu:
	case 0xc000036cu:
		return __tsc(u8"Driver %2 has been blocked from loading.");
	case 0xc000036du:
		return __tsc(u8"There was error [%2] processing the driver database.");
	case 0xc000036eu:
		return __tsc(u8"System hive size has exceeded its limit.");
	case 0xc000036fu:
		return __tsc(u8"A dynamic link library (DLL) referenced a module that was neither a DLL nor the process's executable image.");
	case 0xc0000371u:
		return __tsc(u8"The local account store does not contain secret material for the specified account.");
	case 0xc0000372u:
		return __tsc(u8"Access to %1 has been restricted by your Administrator by policy rule %2.");
	case 0xc0000373u:
		return __tsc(u8"The system was not able to allocate enough memory to perform a stack switch.");
	case 0xc0000374u:
		return __tsc(u8"A heap has been corrupted.");
	case 0xc0000380u:
		return __tsc(u8"An incorrect PIN was presented to the smart card");
	case 0xc0000381u:
		return __tsc(u8"The smart card is blocked");
	case 0xc0000382u:
		return __tsc(u8"No PIN was presented to the smart card");
	case 0xc0000383u:
		return __tsc(u8"No smart card available");
	case 0xc0000384u:
		return __tsc(u8"The requested key container does not exist on the smart card");
	case 0xc0000385u:
		return __tsc(u8"The requested certificate does not exist on the smart card");
	case 0xc0000386u:
		return __tsc(u8"The requested keyset does not exist");
	case 0xc0000387u:
		return __tsc(u8"A communication error with the smart card has been detected.");
	case 0xc0000388u:
		return __tsc(u8"The system cannot contact a domain controller to service the authentication request. Please try again later.");
	case 0xc0000389u:
		return __tsc(u8"The smartcard certificate used for authentication has been revoked. Please contact your system administrator. There may be additional information in the event log.");
	case 0xc000038au:
		return __tsc(u8"An untrusted certificate authority was detected while processing the certificate used for authentication.");
	case 0xc000038bu:
		return __tsc(u8"The revocation status of the certificate used for authentication could not be determined.");
	case 0xc000038cu:
		return __tsc(u8"The client certificate used for authentication was not trusted.");
	case 0xc000038du:
		return __tsc(u8"The smartcard certificate used for authentication has expired. Please\ncontact your system administrator.");
	case 0xc000038eu:
		return __tsc(u8"The driver could not be loaded because a previous version of the driver is still in memory.");
	case 0xc000038fu:
		return __tsc(u8"The smartcard provider could not perform the action since the context was acquired as silent.");
	case 0xc0000401u:
		return __tsc(u8"The current user's delegated trust creation quota has been exceeded.");
	case 0xc0000402u:
		return __tsc(u8"The total delegated trust creation quota has been exceeded.");
	case 0xc0000403u:
		return __tsc(u8"The current user's delegated trust deletion quota has been exceeded.");
	case 0xc0000404u:
		return __tsc(u8"The requested name already exists as a unique identifier.");
	case 0xc0000405u:
		return __tsc(u8"The requested object has a non-unique identifier and cannot be retrieved.");
	case 0xc0000406u:
		return __tsc(u8"The group cannot be converted due to attribute restrictions on the requested group type.");
	case 0xc0000407u:
		return __tsc(u8"{Volume Shadow Copy Service}\nPlease wait while the Volume Shadow Copy Service prepares volume %hs for hibernation.");
	case 0xc0000408u:
		return __tsc(u8"Kerberos sub-protocol User2User is required.");
	case 0xc0000409u:
		return __tsc(u8"The system detected an overrun of a stack-based buffer in this application. This overrun could potentially allow a malicious user to gain control of this application.");
	case 0xc000040au:
		return __tsc(u8"The Kerberos subsystem encountered an error. A service for user protocol request was made against a domain controller which does not support service for user.");
	case 0xc000040bu:
		return __tsc(u8"An attempt was made by this server to make a Kerberos constrained delegation request for a target outside of the server's realm. This is not supported, and indicates a misconfiguration on this server's allowed to delegate to list. Please contact your administrator.");
	case 0xc000040cu:
		return __tsc(u8"The revocation status of the domain controller certificate used for authentication could not be determined. There is additional information in the system event log.");
	case 0xc000040du:
		return __tsc(u8"An untrusted certificate authority was detected while processing the domain controller certificate used for authentication. There is additional information in the system event log. Please contact your system administrator.");
	case 0xc000040eu:
		return __tsc(u8"The domain controller certificate used for logon has expired. There is additional information in the system event log.");
	case 0xc000040fu:
		return __tsc(u8"The domain controller certificate used for logon has been revoked. There is additional information in the system event log.");
	case 0xc0000410u:
		return __tsc(u8"Data present in one of the parameters is more than the function can operate on.");
	case 0xc0000411u:
		return __tsc(u8"The system has failed to hibernate (The error code is %hs). Hibernation will be disabled until the system is restarted.");
	case 0xc0000412u:
		return __tsc(u8"An attempt to delay-load a .dll or get a function address in a delay-loaded .dll failed.");
	case 0xc0000413u:
		return __tsc(u8"Logon Failure: The machine you are logging onto is protected by an authentication firewall. The specified account is not allowed to authenticate to the machine.");
	case 0xc0000414u:
		return __tsc(u8"%hs is a 16-bit application. You do not have permissions to execute 16-bit applications. Check your permissions with your system administrator.");
	case 0xc0000415u:
		return __tsc(u8"{Display Driver Stopped Responding}\nThe %hs display driver has stopped working normally. Save your work and reboot the system to restore full display functionality. The next time you reboot the machine a dialog will be displayed giving you a chance to report this failure to Microsoft.");
	case 0xc0000416u:
		return __tsc(u8"The Desktop heap encountered an error while allocating session memory. There is more information in the system event log.");
	case 0xc0000417u:
		return __tsc(u8"An invalid parameter was passed to a C runtime function.");
	case 0xc0000418u:
		return __tsc(u8"The authentication failed since NTLM was blocked.");
	case 0xc0000419u:
		return __tsc(u8"The source object's SID already exists in destination forest.");
	case 0xc000041au:
		return __tsc(u8"The domain name of the trusted domain already exists in the forest.");
	case 0xc000041bu:
		return __tsc(u8"The flat name of the trusted domain already exists in the forest.");
	case 0xc000041cu:
		return __tsc(u8"The User Principal Name (UPN) is invalid.");
	case 0xc000041du:
		return __tsc(u8"An unhandled exception was encountered during a user callback.");
	case 0xc0000420u:
		return __tsc(u8"An assertion failure has occurred.");
	case 0xc0000421u:
		return __tsc(u8"Application verifier has found an error in the current process.");
	case 0xc0000423u:
		return __tsc(u8"An exception has occurred in a user mode callback and the kernel callback frame should be removed.");
	case 0xc0000424u:
		return __tsc(u8"%2 has been blocked from loading due to incompatibility with this system. Please contact your software vendor for a compatible version of the driver.");
	case 0xc0000425u:
		return __tsc(u8"Illegal operation attempted on a registry key which has already been unloaded.");
	case 0xc0000426u:
		return __tsc(u8"Compression is disabled for this volume.");
	case 0xc0000427u:
		return __tsc(u8"The requested operation could not be completed due to a file system limitation");
	case 0xc0000428u:
		return __tsc(u8"Windows cannot verify the digital signature for this file. A recent hardware or software change might have installed a file that is signed incorrectly or damaged, or that might be malicious software from an unknown source.");
	case 0xc0000429u:
		return __tsc(u8"The implementation is not capable of performing the request.");
	case 0xc000042au:
		return __tsc(u8"The requested operation is out of order with respect to other operations.");
	case 0xc000042bu:
		return __tsc(u8"An operation attempted to exceed an implementation-defined limit.");
	case 0xc000042cu:
		return __tsc(u8"The requested operation requires elevation.");
	case 0xc000042du:
		return __tsc(u8"The required security context does not exist.");
	case 0xc000042fu:
		return __tsc(u8"The PKU2U protocol encountered an error while attempting to utilize the associated certificates.");
	case 0xc0000432u:
		return __tsc(u8"The operation was attempted beyond the valid data length of the file.");
	case 0xc0000433u:
		return __tsc(u8"The attempted write operation encountered a write already in progress for some portion of the range.");
	case 0xc0000434u:
		return __tsc(u8"The page fault mappings changed in the middle of processing a fault so the operation must be retried.");
	case 0xc0000435u:
		return __tsc(u8"The attempt to purge this file from memory failed to purge some or all the data from memory.");
	case 0xc0000440u:
		return __tsc(u8"The requested credential requires confirmation.");
	case 0xc0000441u:
		return __tsc(u8"The remote server sent an invalid response for a file being opened with Client Side Encryption.");
	case 0xc0000442u:
		return __tsc(u8"Client Side Encryption is not supported by the remote server even though it claims to support it.");
	case 0xc0000443u:
		return __tsc(u8"File is encrypted and should be opened in Client Side Encryption mode.");
	case 0xc0000444u:
		return __tsc(u8"A new encrypted file is being created and a $EFS needs to be provided.");
	case 0xc0000445u:
		return __tsc(u8"The SMB client requested a CSE FSCTL on a non-CSE file.");
	case 0xc0000446u:
		return __tsc(u8"Indicates a particular Security ID may not be assigned as the label of an object.");
	case 0xc0000450u:
		return __tsc(u8"The process hosting the driver for this device has terminated.");
	case 0xc0000451u:
		return __tsc(u8"The requested system device cannot be identified due to multiple indistinguishable devices potentially matching the identification criteria.");
	case 0xc0000452u:
		return __tsc(u8"The requested system device cannot be found.");
	case 0xc0000453u:
		return __tsc(u8"This boot application must be restarted.");
	case 0xc0000454u:
		return __tsc(u8"Insufficient NVRAM resources exist to complete the API.  A reboot might be required.");
	case 0xc0000455u:
		return __tsc(u8"The specified session is invalid.");
	case 0xc0000456u:
		return __tsc(u8"The specified thread is already in a session.");
	case 0xc0000457u:
		return __tsc(u8"The specified thread is not in a session.");
	case 0xc0000458u:
		return __tsc(u8"The specified weight is invalid.");
	case 0xc0000459u:
		return __tsc(u8"The operation was paused.");
	case 0xc0000460u:
		return __tsc(u8"No ranges for the specified operation were able to be processed.");
	case 0xc0000461u:
		return __tsc(u8"The physical resources of this disk have been exhausted.");
	case 0xc0000462u:
		return __tsc(u8"The application cannot be started. Try reinstalling the application to fix the problem.");
	case 0xc0000463u:
		return __tsc(u8"{Device Feature Not Supported}\nThe device does not support the command feature.");
	case 0xc0000464u:
		return __tsc(u8"{Source/Destination device unreachable}\nThe device is unreachable.");
	case 0xc0000465u:
		return __tsc(u8"{Invalid Proxy Data Token}\nThe token representing the data is invalid.");
	case 0xc0000466u:
		return __tsc(u8"The file server is temporarily unavailable.");
	case 0xc0000467u:
		return __tsc(u8"The file is temporarily unavailable.");
	case 0xc0000468u:
		return __tsc(u8"{Device Insufficient Resources}\nThe target device has insufficient resources to complete the operation.");
	case 0xc0000469u:
		return __tsc(u8"The application cannot be started because it is currently updating.");
	case 0xc000046au:
		return __tsc(u8"The specified copy of the requested data could not be read.");
	case 0xc000046bu:
		return __tsc(u8"The specified data could not be written to any of the copies.");
	case 0xc000046cu:
		return __tsc(u8"One or more copies of data on this device may be out of sync. No writes may be performed until a data integrity scan is completed.");
	case 0xc000046du:
		return __tsc(u8"This object is not externally backed by any provider.");
	case 0xc000046eu:
		return __tsc(u8"The external backing provider is not recognized.");
	case 0xc000046fu:
		return __tsc(u8"Compressing this object would not save space.");
	case 0xc0000470u:
		return __tsc(u8"A data integrity checksum error occurred. Data in the file stream is corrupt.");
	case 0xc0000471u:
		return __tsc(u8"An attempt was made to modify both a KERNEL and normal Extended Attribute (EA) in the same operation.");
	case 0xc0000472u:
		return __tsc(u8"{LogicalBlockProvisioningReadZero Not Supported}\nThe target device does not support read returning zeros from trimmed/unmapped blocks.");
	case 0xc0000473u:
		return __tsc(u8"{Maximum Segment Descriptors Exceeded}\nThe command specified a number of descriptors that exceeded the maximum supported by the device.");
	case 0xc0000474u:
		return __tsc(u8"{Alignment Violation}\nThe command specified a data offset that does not align to the device's granularity/alignment.");
	case 0xc0000475u:
		return __tsc(u8"{Invalid Field In Parameter List}\nThe command specified an invalid field in its parameter list.");
	case 0xc0000476u:
		return __tsc(u8"{Operation In Progress}\nAn operation is currently in progress with the device.");
	case 0xc0000477u:
		return __tsc(u8"{Invalid I_T Nexus}\nAn attempt was made to send down the command via an invalid path to the target device.");
	case 0xc0000478u:
		return __tsc(u8"Scrub is disabled on the specified file.");
	case 0xc0000479u:
		return __tsc(u8"The storage device does not provide redundancy.");
	case 0xc000047au:
		return __tsc(u8"An operation is not supported on a resident file.");
	case 0xc000047bu:
		return __tsc(u8"An operation is not supported on a compressed file.");
	case 0xc000047cu:
		return __tsc(u8"An operation is not supported on a directory.");
	case 0xc000047du:
		return __tsc(u8"{IO Operation Timeout}\nThe specified I/O operation failed to complete within the expected time period.");
	case 0xc000047eu:
		return __tsc(u8"An error in a system binary was detected. Try refreshing the PC to fix the problem.");
	case 0xc000047fu:
		return __tsc(u8"A corrupted CLR NGEN binary was detected on the system.");
	case 0xc0000480u:
		return __tsc(u8"The share is temporarily unavailable.");
	case 0xc0000481u:
		return __tsc(u8"The target dll was not found because the apiset %hs is not hosted.");
	case 0xc0000482u:
		return __tsc(u8"The API set extension contains a host for a non-existent API set.");
	case 0xc0000483u:
		return __tsc(u8"The request failed due to a fatal device hardware error.");
	case 0xc0000484u:
		return __tsc(u8"The specified firmware slot is invalid.");
	case 0xc0000485u:
		return __tsc(u8"The specified firmware image is invalid.");
	case 0xc0000486u:
		return __tsc(u8"The request failed due to a storage topology ID mismatch.");
	case 0xc0000487u:
		return __tsc(u8"The specified Windows Image (WIM) is not marked as bootable.");
	case 0xc0000488u:
		return __tsc(u8"The operation was blocked by parental controls.");
	case 0xc0000489u:
		return __tsc(u8"The deployment operation failed because the specified application needs to be registered first.");
	case 0xc000048au:
		return __tsc(u8"The requested operation failed due to quota operation is still in progress.");
	case 0xc000048bu:
		return __tsc(u8"The callback function must be invoked inline.");
	case 0xc000048cu:
		return __tsc(u8"A file system block being referenced has already reached the maximum reference count and can't be referenced any further.");
	case 0xc000048du:
		return __tsc(u8"The requested operation failed because the file stream is marked to disallow writes.");
	case 0xc000048eu:
		return __tsc(u8"Windows Information Protection policy does not allow access to this network resource.");
	case 0xc000048fu:
		return __tsc(u8"The requested operation failed with an architecture-specific failure code.");
	case 0xc0000490u:
		return __tsc(u8"There are no compatible drivers available for this device.");
	case 0xc0000491u:
		return __tsc(u8"The specified driver package cannot be found on the system.");
	case 0xc0000492u:
		return __tsc(u8"The driver package cannot find a required driver configuration.");
	case 0xc0000493u:
		return __tsc(u8"The driver configuration is incomplete for use with this device.");
	case 0xc0000494u:
		return __tsc(u8"The device requires a driver configuration with a function driver.");
	case 0xc0000495u:
		return __tsc(u8"The device is pending further configuration.");
	case 0xc0000496u:
		return __tsc(u8"The device hint name buffer is too small to receive the remaining name.");
	case 0xc0000497u:
		return __tsc(u8"The package is currently not available.");
	case 0xc0000499u:
		return __tsc(u8"The device is in maintenance mode.");
	case 0xc000049au:
		return __tsc(u8"This operation is not supported on a DAX volume.");
	case 0xc000049bu:
		return __tsc(u8"The free space on the volume is too fragmented to complete this operation.");
	case 0xc000049cu:
		return __tsc(u8"The volume has active DAX mappings.");
	case 0xc000049du:
		return __tsc(u8"The process creation has been blocked.");
	case 0xc000049eu:
		return __tsc(u8"The storage device has lost data or persistence.");
	case 0xc000049fu:
		return __tsc(u8"Driver Verifier Volatile settings cannot be set when CFG is enabled.");
	case 0xc00004a0u:
	case 0xc00004a1u:
	case 0xc00004a2u:
	case 0xc00004a3u:
	case 0xc00004a4u:
	case 0xc00004a5u:
	case 0xc00004a6u:
	case 0xc00004a7u:
	case 0xc00004a8u:
	case 0xc00004a9u:
	case 0xc00004aau:
	case 0xc00004abu:
	case 0xc00004acu:
	case 0xc00004adu:
	case 0xc00004aeu:
	case 0xc00004afu:
	case 0xc00004b0u:
	case 0xc00004b1u:
	case 0xc00004b2u:
	case 0xc00004b3u:
	case 0xc00004b4u:
	case 0xc00004b5u:
	case 0xc00004b6u:
	case 0xc00004b7u:
	case 0xc00004b8u:
	case 0xc00004b9u:
	case 0xc00004bau:
	case 0xc00004bbu:
	case 0xc00004bcu:
	case 0xc00004bdu:
	case 0xc00004beu:
	case 0xc00004bfu:
	case 0xc00004c0u:
	case 0xc00004c1u:
	case 0xc00004c2u:
	case 0xc00004c3u:
	case 0xc00004c4u:
	case 0xc00004c5u:
	case 0xc00004c6u:
	case 0xc00004c7u:
	case 0xc00004c8u:
	case 0xc00004c9u:
	case 0xc00004cau:
	case 0xc00004cbu:
	case 0xc00004ccu:
	case 0xc00004cdu:
	case 0xc00004ceu:
	case 0xc00004cfu:
	case 0xc00004d0u:
	case 0xc00004d1u:
	case 0xc00004d2u:
	case 0xc00004d3u:
	case 0xc00004d4u:
	case 0xc00004d5u:
		return __tsc(u8"");
	case 0xc0000500u:
		return __tsc(u8"The specified task name is invalid.");
	case 0xc0000501u:
		return __tsc(u8"The specified task index is invalid.");
	case 0xc0000502u:
		return __tsc(u8"The specified thread is already joining a task.");
	case 0xc0000503u:
		return __tsc(u8"A callback has requested to bypass native code.");
	case 0xc0000504u:
		return __tsc(u8"The Central Access Policy specified is not defined on the target machine.");
	case 0xc0000505u:
		return __tsc(u8"The Central Access Policy obtained from Active Directory is invalid.");
	case 0xc0000506u:
		return __tsc(u8"Unable to finish the requested operation because the specified process is not a GUI process.");
	case 0xc0000507u:
		return __tsc(u8"The device is not responding and cannot be safely removed.");
	case 0xc0000508u:
		return __tsc(u8"The specified Job already has a container assigned to it.");
	case 0xc0000509u:
		return __tsc(u8"The specified Job does not have a container assigned to it.");
	case 0xc000050au:
		return __tsc(u8"The device is unresponsive.");
	case 0xc000050bu:
		return __tsc(u8"The object manager encountered a reparse point while retrieving an object.");
	case 0xc000050cu:
		return __tsc(u8"The requested attribute is not present on the specified file or directory.");
	case 0xc000050du:
		return __tsc(u8"This volume is not a tiered volume.");
	case 0xc000050eu:
		return __tsc(u8"This file is currently associated with a different stream id.");
	case 0xc000050fu:
		return __tsc(u8"The requested operation could not be completed because the specified job has children.");
	case 0xc0000510u:
		return __tsc(u8"The specified object has already been initialized.");
	case 0xc0000511u:
	case 0xc0000512u:
	case 0xc0000513u:
	case 0xc0000514u:
	case 0xc0000515u:
	case 0xc0000516u:
	case 0xc0000517u:
	case 0xc0000518u:
		return __tsc(u8"");
	case 0xc0000602u:
		return __tsc(u8"{Fail Fast Exception}\nA fail fast exception occurred. Exception handlers will not be invoked and the process will be terminated immediately.");
	case 0xc0000603u:
		return __tsc(u8"Windows cannot verify the digital signature for this file. The signing certificate for this file has been revoked.");
	case 0xc0000604u:
		return __tsc(u8"The operation was blocked as the process prohibits dynamic code generation.");
	case 0xc0000605u:
		return __tsc(u8"Windows cannot verify the digital signature for this file. The signing certificate for this file has expired.");
	case 0xc0000606u:
		return __tsc(u8"The specified image file was blocked from loading because it does not enable a feature required by the process: Control Flow Guard.");
	case 0xc000060au:
		return __tsc(u8"The thread context could not be updated because this has been restricted for the process.");
	case 0xc000060bu:
		return __tsc(u8"An attempt to access another partition's private file/section was rejected.");
	case 0xc0000700u:
		return __tsc(u8"The ALPC port is closed.");
	case 0xc0000701u:
		return __tsc(u8"The ALPC message requested is no longer available.");
	case 0xc0000702u:
		return __tsc(u8"The ALPC message supplied is invalid.");
	case 0xc0000703u:
		return __tsc(u8"The ALPC message has been canceled.");
	case 0xc0000704u:
		return __tsc(u8"Invalid recursive dispatch attempt.");
	case 0xc0000705u:
		return __tsc(u8"No receive buffer has been supplied in a synchrounus request.");
	case 0xc0000706u:
		return __tsc(u8"The connection port is used in an invalid context.");
	case 0xc0000707u:
		return __tsc(u8"The ALPC port does not accept new request messages.");
	case 0xc0000708u:
		return __tsc(u8"The resource requested is already in use.");
	case 0xc0000709u:
		return __tsc(u8"The hardware has reported an uncorrectable memory error.");
	case 0xc000070au:
		return __tsc(u8"Status 0x");
	case 0xc000070bu:
		return __tsc(u8"After a callback to 0x%p(0x%p), a completion call to SetEvent(0x%p) failed with status 0x");
	case 0xc000070cu:
		return __tsc(u8"After a callback to 0x%p(0x%p), a completion call to ReleaseSemaphore(0x%p, %d) failed with status 0x");
	case 0xc000070du:
		return __tsc(u8"After a callback to 0x%p(0x%p), a completion call to ReleaseMutex(%p) failed with status 0x");
	case 0xc000070eu:
		return __tsc(u8"After a callback to 0x%p(0x%p), an completion call to FreeLibrary(%p) failed with status 0x");
	case 0xc000070fu:
		return __tsc(u8"The threadpool 0x%p was released while a thread was posting a callback to 0x%p(0x%p) to it.");
	case 0xc0000710u:
		return __tsc(u8"A threadpool worker thread is impersonating a client, after a callback to 0x%p(0x%p).\nThis is unexpected, indicating that the callback is missing a call to revert the impersonation.");
	case 0xc0000711u:
		return __tsc(u8"A threadpool worker thread is impersonating a client, after executing an APC.\nThis is unexpected, indicating that the APC is missing a call to revert the impersonation.");
	case 0xc0000712u:
		return __tsc(u8"Either the target process, or the target thread's containing process, is a protected process.");
	case 0xc0000713u:
		return __tsc(u8"A Thread is getting dispatched with MCA EXCEPTION because of MCA.");
	case 0xc0000714u:
		return __tsc(u8"The client certificate account mapping is not unique.");
	case 0xc0000715u:
		return __tsc(u8"The symbolic link cannot be followed because its type is disabled.");
	case 0xc0000716u:
		return __tsc(u8"Indicates that the specified string is not valid for IDN normalization.");
	case 0xc0000717u:
		return __tsc(u8"No mapping for the Unicode character exists in the target multi-byte code page.");
	case 0xc0000718u:
		return __tsc(u8"The provided callback is already registered.");
	case 0xc0000719u:
		return __tsc(u8"The provided context did not match the target.");
	case 0xc000071au:
		return __tsc(u8"The specified port already has a completion list.");
	case 0xc000071bu:
		return __tsc(u8"A threadpool worker thread enter a callback at thread base priority 0x%x and exited at priority 0x%x.\nThis is unexpected, indicating that the callback missed restoring the priority.");
	case 0xc000071cu:
		return __tsc(u8"An invalid thread, handle %p, is specified for this operation. Possibly, a threadpool worker thread was specified.");
	case 0xc000071du:
		return __tsc(u8"A threadpool worker thread enter a callback, which left transaction state.\nThis is unexpected, indicating that the callback missed clearing the transaction.");
	case 0xc000071eu:
		return __tsc(u8"A threadpool worker thread enter a callback, which left the loader lock held.\nThis is unexpected, indicating that the callback missed releasing the lock.");
	case 0xc000071fu:
		return __tsc(u8"A threadpool worker thread enter a callback, which left with preferred languages set.\nThis is unexpected, indicating that the callback missed clearing them.");
	case 0xc0000720u:
		return __tsc(u8"A threadpool worker thread enter a callback, which left with background priorities set.\nThis is unexpected, indicating that the callback missed restoring the original priorities.");
	case 0xc0000721u:
		return __tsc(u8"A threadpool worker thread enter a callback at thread affinity %p and exited at affinity %p.\nThis is unexpected, indicating that the callback missed restoring the priority.");
	case 0xc0000722u:
		return __tsc(u8"The caller has exceeded the maximum number of handles that may be transmitted in\na single local procedure call.");
	case 0xc0000723u:
	case 0xc0000724u:
	case 0xc0000725u:
	case 0xc0000726u:
		return __tsc(u8"");
	case 0xc0000800u:
		return __tsc(u8"The attempted operation required self healing to be enabled.");
	case 0xc0000801u:
		return __tsc(u8"The Directory Service cannot perform the requested operation because a domain rename operation is in progress.");
	case 0xc0000802u:
		return __tsc(u8"The requested file operation failed because the storage quota was exceeded.\nTo free up disk space, move files to a different location or delete unnecessary files. For more information, contact your system administrator.");
	case 0xc0000804u:
		return __tsc(u8"The requested file operation failed because the storage policy blocks that type of file. For more information, contact your system administrator.");
	case 0xc0000805u:
		return __tsc(u8"The operation could not be completed due to bad clusters on disk.");
	case 0xc0000806u:
		return __tsc(u8"The operation could not be completed because the volume is dirty. Please run chkdsk and try again.");
	case 0xc0000808u:
		return __tsc(u8"The volume repair was not successful.");
	case 0xc0000809u:
		return __tsc(u8"One of the volume corruption logs is full. Further corruptions that may be detected won't be logged.");
	case 0xc000080au:
		return __tsc(u8"One of the volume corruption logs is internally corrupted and needs to be recreated. The volume may contain undetected corruptions and must be scanned.");
	case 0xc000080bu:
		return __tsc(u8"One of the volume corruption logs is unavailable for being operated on.");
	case 0xc000080cu:
		return __tsc(u8"One of the volume corruption logs was deleted while still having corruption records in them. The volume contains detected corruptions and must be scanned.");
	case 0xc000080du:
		return __tsc(u8"One of the volume corruption logs was cleared by chkdsk and no longer contains real corruptions.");
	case 0xc000080eu:
		return __tsc(u8"Orphaned files exist on the volume but could not be recovered because no more new names could be created in the recovery directory. Files must be moved from the recovery directory.");
	case 0xc000080fu:
		return __tsc(u8"The operation could not be completed because an instance of Proactive Scanner is currently running.");
	case 0xc0000810u:
		return __tsc(u8"The read or write operation to an encrypted file could not be completed because the file has not been opened for data access.");
	case 0xc0000811u:
		return __tsc(u8"One of the volume corruption logs comes from a newer version of Windows and contains corruption records. The log will be emptied and reset to the current version, and the volume health state will be updated accordingly.");
	case 0xc0000901u:
		return __tsc(u8"This file is checked out or locked for editing by another user.");
	case 0xc0000902u:
		return __tsc(u8"The file must be checked out before saving changes.");
	case 0xc0000903u:
		return __tsc(u8"The file type being saved or retrieved has been blocked.");
	case 0xc0000904u:
		return __tsc(u8"The file size exceeds the limit allowed and cannot be saved.");
	case 0xc0000905u:
		return __tsc(u8"Access Denied. Before opening files in this location, you must first browse to the web site and select the option to login automatically.");
	case 0xc0000906u:
		return __tsc(u8"Operation did not complete successfully because the file contains a virus or potentially unwanted software.");
	case 0xc0000907u:
		return __tsc(u8"This file contains a virus or potentially unwanted software and cannot be opened. Due to the nature of this virus or potentially unwanted software, the file has been removed from this location.");
	case 0xc0000908u:
		return __tsc(u8"The resources required for this device conflict with the MCFG table.");
	case 0xc0000909u:
		return __tsc(u8"The operation did not complete successfully because it would cause an oplock to be broken. The caller has requested that existing oplocks not be broken.");
	case 0xc000090au:
		return __tsc(u8"Bad key.");
	case 0xc000090bu:
		return __tsc(u8"Bad data.");
	case 0xc000090cu:
		return __tsc(u8"Key does not exist.");
	case 0xc0000910u:
		return __tsc(u8"Access to the specified file handle has been revoked.");
	case 0xc0000911u:
	case 0xc0000912u:
	case 0xc0000913u:
	case 0xc0000914u:
	case 0xc0000915u:
	case 0xc0000c08u:
	case 0xc0000c09u:
	case 0xc0000c0au:
	case 0xc0000c0bu:
	case 0xc0000c0cu:
	case 0xc0000c0du:
	case 0xc0000c0eu:
	case 0xc0000c0fu:
	case 0xc0000c76u:
	case 0xc0000c77u:
	case 0xc0000c78u:
	case 0xc0000c79u:
	case 0xc0000c7au:
	case 0xc0000c7bu:
	case 0xc0000c7cu:
	case 0xc0000c7du:
	case 0xc0000c7eu:
	case 0xc0000c7fu:
		return __tsc(u8"");
	case 0xc0009898u:
		return __tsc(u8"WOW Assertion Error.");
	case 0xc000a000u:
		return __tsc(u8"The cryptographic signature is invalid.");
	case 0xc000a001u:
		return __tsc(u8"The cryptographic provider does not support HMAC.");
	case 0xc000a002u:
		return __tsc(u8"The computed authentication tag did not match the input authentication tag.");
	case 0xc000a003u:
		return __tsc(u8"The requested state transition is invalid and cannot be performed.");
	case 0xc000a004u:
		return __tsc(u8"The supplied kernel information version is invalid.");
	case 0xc000a005u:
		return __tsc(u8"The supplied PEP information version is invalid.");
	case 0xc000a006u:
		return __tsc(u8"Access to the specified handle has been revoked.");
	case 0xc000a007u:
		return __tsc(u8"The file operation will result in the end of file being on a ghosted range.");
	case 0xc000a008u:
		return __tsc(u8"");
	case 0xc000a010u:
		return __tsc(u8"The IPSEC queue overflowed.");
	case 0xc000a011u:
		return __tsc(u8"The neighbor discovery queue overflowed.");
	case 0xc000a012u:
		return __tsc(u8"An ICMP hop limit exceeded error was received.");
	case 0xc000a013u:
		return __tsc(u8"The protocol is not installed on the local machine.");
	case 0xc000a014u:
		return __tsc(u8"An operation or data has been rejected while on the network fast path.");
	case 0xc000a080u:
		return __tsc(u8"{Delayed Write Failed}\nWindows was unable to save all the data for the file %hs; the data has been lost.\nThis error may be caused by network connectivity issues. Please try to save this file elsewhere.");
	case 0xc000a081u:
		return __tsc(u8"{Delayed Write Failed}\nWindows was unable to save all the data for the file %hs; the data has been lost.\nThis error was returned by the server on which the file exists. Please try to save this file elsewhere.");
	case 0xc000a082u:
		return __tsc(u8"{Delayed Write Failed}\nWindows was unable to save all the data for the file %hs; the data has been lost.\nThis error may be caused if the device has been removed or the media is write-protected.");
	case 0xc000a083u:
		return __tsc(u8"Windows was unable to parse the requested XML data.");
	case 0xc000a084u:
		return __tsc(u8"An error was encountered while processing an XML digital signature.");
	case 0xc000a085u:
		return __tsc(u8"Indicates that the caller made the connection request in the wrong routing compartment.");
	case 0xc000a086u:
		return __tsc(u8"Indicates that there was an AuthIP failure when attempting to connect to the remote host.");
	case 0xc000a087u:
		return __tsc(u8"OID mapped groups cannot have members.");
	case 0xc000a088u:
		return __tsc(u8"The specified OID cannot be found.");
	case 0xc000a089u:
		return __tsc(u8"The system is not authoritative for the specified account and therefore cannot complete the operation. Please retry the operation using the provider associated with this account. If this is an online provider please use the provider's online site.");
	case 0xc000a08au:
	case 0xc000a08bu:
	case 0xc000a08cu:
	case 0xc000a08du:
	case 0xc000a08eu:
		return __tsc(u8"");
	case 0xc000a100u:
		return __tsc(u8"Hash generation for the specified version and hash type is not enabled on server.");
	case 0xc000a101u:
		return __tsc(u8"The hash requests is not present or not up to date with the current file contents.");
	case 0xc000a121u:
		return __tsc(u8"The secondary interrupt controller instance that manages the specified interrupt is not registered.");
	case 0xc000a122u:
		return __tsc(u8"The information supplied by the GPIO client driver is invalid.");
	case 0xc000a123u:
		return __tsc(u8"The version specified by the GPIO client driver is not supported.");
	case 0xc000a124u:
		return __tsc(u8"The registration packet supplied by the GPIO client driver is not valid.");
	case 0xc000a125u:
		return __tsc(u8"The requested operation is not suppported for the specified handle.");
	case 0xc000a126u:
		return __tsc(u8"The requested connect mode conflicts with an existing mode on one or more of the specified pins.");
	case 0xc000a141u:
		return __tsc(u8"The requested run level switch cannot be completed successfully since\none or more services refused to stop or restart.");
	case 0xc000a142u:
		return __tsc(u8"The service has an invalid run level setting. The run level for a service\nmust not be higher than the run level of its dependent services.");
	case 0xc000a143u:
		return __tsc(u8"The requested run level switch cannot be completed successfully since\none or more services will not stop or restart within the specified timeout.");
	case 0xc000a145u:
		return __tsc(u8"A run level switch agent did not respond within the specified timeout.");
	case 0xc000a146u:
		return __tsc(u8"A run level switch is currently in progress.");
	case 0xc000a200u:
		return __tsc(u8"This operation is only valid in the context of an app container.");
	case 0xc000a201u:
		return __tsc(u8"This functionality is not supported in the context of an app container.");
	case 0xc000a202u:
		return __tsc(u8"The length of the SID supplied is not a valid length for app container SIDs.");
	case 0xc000a203u:
		return __tsc(u8"Access to the specified resource has been denied for a less privileged app container.");
	case 0xc000a204u:
		return __tsc(u8"");
	case 0xc000a281u:
		return __tsc(u8"Fast Cache data not found.");
	case 0xc000a282u:
		return __tsc(u8"Fast Cache data expired.");
	case 0xc000a283u:
		return __tsc(u8"Fast Cache data corrupt.");
	case 0xc000a284u:
		return __tsc(u8"Fast Cache data has exceeded its max size and cannot be updated.");
	case 0xc000a285u:
		return __tsc(u8"Fast Cache has been ReArmed and requires a reboot until it can be updated.");
	case 0xc000a2a1u:
		return __tsc(u8"The copy offload read operation is not supported by a filter.");
	case 0xc000a2a2u:
		return __tsc(u8"The copy offload write operation is not supported by a filter.");
	case 0xc000a2a3u:
		return __tsc(u8"The copy offload read operation is not supported for the file.");
	case 0xc000a2a4u:
		return __tsc(u8"The copy offload write operation is not supported for the file.");
	case 0xc000a2a5u:
	case 0xc000a2a6u:
	case 0xc000a2a7u:
	case 0xc000c001u:
	case 0xc000c002u:
		return __tsc(u8"");
	case 0xc000ce01u:
		return __tsc(u8"The provider that supports file system virtualization is temporarily unavailable.");
	case 0xc000ce02u:
		return __tsc(u8"The metadata for file system virtualization is corrupt and unreadable.");
	case 0xc000ce03u:
		return __tsc(u8"The provider that supports file system virtualization is too busy to complete this operation.");
	case 0xc000ce04u:
		return __tsc(u8"The provider that supports file system virtualization is unknown.");
	case 0xc000ce05u:
		return __tsc(u8"");
	case 0xc000cf00u:
		return __tsc(u8"The Cloud File provider is unknown.");
	case 0xc000cf01u:
		return __tsc(u8"The Cloud File provider is not running.");
	case 0xc000cf02u:
		return __tsc(u8"The Cloud File metadata is corrupt and unreadable.");
	case 0xc000cf03u:
		return __tsc(u8"The operation could not be completed because the Cloud File metadata is too large.");
	case 0xc000cf06u:
		return __tsc(u8"The operation could not be completed because the Cloud File metadata version is not supported.");
	case 0xc000cf07u:
		return __tsc(u8"The operation could not be completed because the file is not a Cloud File.");
	case 0xc000cf08u:
		return __tsc(u8"The operation could not be completed because the Cloud File is not in sync.");
	case 0xc000cf09u:
	case 0xc000cf0au:
	case 0xc000cf0bu:
	case 0xc000cf0cu:
	case 0xc000cf0du:
	case 0xc000cf0eu:
	case 0xc000cf0fu:
	case 0xc000cf10u:
	case 0xc000cf11u:
	case 0xc000cf12u:
	case 0xc000cf13u:
	case 0xc000cf14u:
	case 0xc000cf15u:
	case 0xc000cf16u:
	case 0xc000cf17u:
	case 0xc000cf18u:
	case 0xc000cf19u:
	case 0xc000cf1au:
	case 0xc000cf1bu:
	case 0xc000cf1du:
	case 0xc000cf1eu:
	case 0xc000cf1fu:
	case 0xc000cf20u:
	case 0xc000cf21u:
	case 0xc000f500u:
	case 0xc000f501u:
	case 0xc000f502u:
	case 0xc000f503u:
	case 0xc000f504u:
	case 0xc000f505u:
		return __tsc(u8"");
	case 0xc0010001u:
		return __tsc(u8"The debugger did not perform a state change.");
	case 0xc0010002u:
		return __tsc(u8"The debugger found that the application is not idle.");
	case 0xc0020001u:
		return __tsc(u8"The string binding is invalid.");
	case 0xc0020002u:
		return __tsc(u8"The binding handle is not the correct type.");
	case 0xc0020003u:
		return __tsc(u8"The binding handle is invalid.");
	case 0xc0020004u:
		return __tsc(u8"The RPC protocol sequence is not supported.");
	case 0xc0020005u:
		return __tsc(u8"The RPC protocol sequence is invalid.");
	case 0xc0020006u:
		return __tsc(u8"The string UUID is invalid.");
	case 0xc0020007u:
		return __tsc(u8"The endpoint format is invalid.");
	case 0xc0020008u:
		return __tsc(u8"The network address is invalid.");
	case 0xc0020009u:
		return __tsc(u8"No endpoint was found.");
	case 0xc002000au:
		return __tsc(u8"The time-out value is invalid.");
	case 0xc002000bu:
		return __tsc(u8"The object UUID was not found.");
	case 0xc002000cu:
		return __tsc(u8"The object UUID has already been registered.");
	case 0xc002000du:
		return __tsc(u8"The type UUID has already been registered.");
	case 0xc002000eu:
		return __tsc(u8"The RPC server is already listening.");
	case 0xc002000fu:
		return __tsc(u8"No protocol sequences have been registered.");
	case 0xc0020010u:
		return __tsc(u8"The RPC server is not listening.");
	case 0xc0020011u:
		return __tsc(u8"The manager type is unknown.");
	case 0xc0020012u:
		return __tsc(u8"The interface is unknown.");
	case 0xc0020013u:
		return __tsc(u8"There are no bindings.");
	case 0xc0020014u:
		return __tsc(u8"There are no protocol sequences.");
	case 0xc0020015u:
		return __tsc(u8"The endpoint cannot be created.");
	case 0xc0020016u:
		return __tsc(u8"Insufficient resources are available to complete this operation.");
	case 0xc0020017u:
		return __tsc(u8"The RPC server is unavailable.");
	case 0xc0020018u:
		return __tsc(u8"The RPC server is too busy to complete this operation.");
	case 0xc0020019u:
		return __tsc(u8"The network options are invalid.");
	case 0xc002001au:
		return __tsc(u8"No RPCs are active on this thread.");
	case 0xc002001bu:
		return __tsc(u8"The RPC failed.");
	case 0xc002001cu:
		return __tsc(u8"The RPC failed and did not execute.");
	case 0xc002001du:
		return __tsc(u8"An RPC protocol error occurred.");
	case 0xc002001fu:
		return __tsc(u8"The RPC server does not support the transfer syntax.");
	case 0xc0020021u:
		return __tsc(u8"The type UUID is not supported.");
	case 0xc0020022u:
		return __tsc(u8"The tag is invalid.");
	case 0xc0020023u:
		return __tsc(u8"The array bounds are invalid.");
	case 0xc0020024u:
		return __tsc(u8"The binding does not contain an entry name.");
	case 0xc0020025u:
		return __tsc(u8"The name syntax is invalid.");
	case 0xc0020026u:
		return __tsc(u8"The name syntax is not supported.");
	case 0xc0020028u:
		return __tsc(u8"No network address is available to construct a UUID.");
	case 0xc0020029u:
		return __tsc(u8"The endpoint is a duplicate.");
	case 0xc002002au:
		return __tsc(u8"The authentication type is unknown.");
	case 0xc002002bu:
		return __tsc(u8"The maximum number of calls is too small.");
	case 0xc002002cu:
		return __tsc(u8"The string is too long.");
	case 0xc002002du:
		return __tsc(u8"The RPC protocol sequence was not found.");
	case 0xc002002eu:
		return __tsc(u8"The procedure number is out of range.");
	case 0xc002002fu:
		return __tsc(u8"The binding does not contain any authentication information.");
	case 0xc0020030u:
		return __tsc(u8"The authentication service is unknown.");
	case 0xc0020031u:
		return __tsc(u8"The authentication level is unknown.");
	case 0xc0020032u:
		return __tsc(u8"The security context is invalid.");
	case 0xc0020033u:
		return __tsc(u8"The authorization service is unknown.");
	case 0xc0020034u:
		return __tsc(u8"The entry is invalid.");
	case 0xc0020035u:
		return __tsc(u8"The operation cannot be performed.");
	case 0xc0020036u:
		return __tsc(u8"No more endpoints are available from the endpoint mapper.");
	case 0xc0020037u:
		return __tsc(u8"No interfaces have been exported.");
	case 0xc0020038u:
		return __tsc(u8"The entry name is incomplete.");
	case 0xc0020039u:
		return __tsc(u8"The version option is invalid.");
	case 0xc002003au:
		return __tsc(u8"There are no more members.");
	case 0xc002003bu:
		return __tsc(u8"There is nothing to unexport.");
	case 0xc002003cu:
		return __tsc(u8"The interface was not found.");
	case 0xc002003du:
		return __tsc(u8"The entry already exists.");
	case 0xc002003eu:
		return __tsc(u8"The entry was not found.");
	case 0xc002003fu:
		return __tsc(u8"The name service is unavailable.");
	case 0xc0020040u:
		return __tsc(u8"The network address family is invalid.");
	case 0xc0020041u:
		return __tsc(u8"The requested operation is not supported.");
	case 0xc0020042u:
		return __tsc(u8"No security context is available to allow impersonation.");
	case 0xc0020043u:
		return __tsc(u8"An internal error occurred in the RPC.");
	case 0xc0020044u:
		return __tsc(u8"The RPC server attempted to divide an integer by zero.");
	case 0xc0020045u:
		return __tsc(u8"An addressing error occurred in the RPC server.");
	case 0xc0020046u:
		return __tsc(u8"A floating point operation at the RPC server caused a divide by zero.");
	case 0xc0020047u:
		return __tsc(u8"A floating point underflow occurred at the RPC server.");
	case 0xc0020048u:
		return __tsc(u8"A floating point overflow occurred at the RPC server.");
	case 0xc0020049u:
		return __tsc(u8"An RPC is already in progress for this thread.");
	case 0xc002004au:
		return __tsc(u8"There are no more bindings.");
	case 0xc002004bu:
		return __tsc(u8"The group member was not found.");
	case 0xc002004cu:
		return __tsc(u8"The endpoint mapper database entry could not be created.");
	case 0xc002004du:
		return __tsc(u8"The object UUID is the nil UUID.");
	case 0xc002004fu:
		return __tsc(u8"No interfaces have been registered.");
	case 0xc0020050u:
		return __tsc(u8"The RPC was canceled.");
	case 0xc0020051u:
		return __tsc(u8"The binding handle does not contain all the required information.");
	case 0xc0020052u:
		return __tsc(u8"A communications failure occurred during an RPC.");
	case 0xc0020053u:
		return __tsc(u8"The requested authentication level is not supported.");
	case 0xc0020054u:
		return __tsc(u8"No principal name was registered.");
	case 0xc0020055u:
		return __tsc(u8"The error specified is not a valid Windows RPC error code.");
	case 0xc0020057u:
		return __tsc(u8"A security package-specific error occurred.");
	case 0xc0020058u:
		return __tsc(u8"The thread was not canceled.");
	case 0xc0020062u:
		return __tsc(u8"Invalid asynchronous RPC handle.");
	case 0xc0020063u:
		return __tsc(u8"Invalid asynchronous RPC call handle for this operation.");
	case 0xc0020064u:
		return __tsc(u8"Access to the HTTP proxy is denied.");
	case 0xc0020065u:
		return __tsc(u8"");
	case 0xc0030001u:
		return __tsc(u8"The list of RPC servers available for auto-handle binding has been exhausted.");
	case 0xc0030002u:
		return __tsc(u8"The file designated by DCERPCCHARTRANS cannot be opened.");
	case 0xc0030003u:
		return __tsc(u8"The file containing the character translation table has fewer than 512 bytes.");
	case 0xc0030004u:
		return __tsc(u8"A null context handle is passed as an [in] parameter.");
	case 0xc0030005u:
		return __tsc(u8"The context handle does not match any known context handles.");
	case 0xc0030006u:
		return __tsc(u8"The context handle changed during a call.");
	case 0xc0030007u:
		return __tsc(u8"The binding handles passed to an RPC do not match.");
	case 0xc0030008u:
		return __tsc(u8"The stub is unable to get the call handle.");
	case 0xc0030009u:
		return __tsc(u8"A null reference pointer was passed to the stub.");
	case 0xc003000au:
		return __tsc(u8"The enumeration value is out of range.");
	case 0xc003000bu:
		return __tsc(u8"The byte count is too small.");
	case 0xc003000cu:
		return __tsc(u8"The stub received bad data.");
	case 0xc0030059u:
		return __tsc(u8"Invalid operation on the encoding/decoding handle.");
	case 0xc003005au:
		return __tsc(u8"Incompatible version of the serializing package.");
	case 0xc003005bu:
		return __tsc(u8"Incompatible version of the RPC stub.");
	case 0xc003005cu:
		return __tsc(u8"The RPC pipe object is invalid or corrupt.");
	case 0xc003005du:
		return __tsc(u8"An invalid operation was attempted on an RPC pipe object.");
	case 0xc003005eu:
		return __tsc(u8"Unsupported RPC pipe version.");
	case 0xc003005fu:
		return __tsc(u8"The RPC pipe object has already been closed.");
	case 0xc0030060u:
		return __tsc(u8"The RPC call completed before all pipes were processed.");
	case 0xc0030061u:
		return __tsc(u8"No more data is available from the RPC pipe.");
	case 0xc0040035u:
		return __tsc(u8"A device is missing in the system BIOS MPS table. This device will not be used. Contact your system vendor for a system BIOS update.");
	case 0xc0040036u:
		return __tsc(u8"A translator failed to translate resources.");
	case 0xc0040037u:
		return __tsc(u8"An IRQ translator failed to translate resources.");
	case 0xc0040038u:
		return __tsc(u8"Driver %2 returned an invalid ID for a child device (%3).");
	case 0xc0040039u:
		return __tsc(u8"Reissue the given operation as a cached I/O operation");
	case 0xc00a0001u:
		return __tsc(u8"Session name %1 is invalid.");
	case 0xc00a0002u:
		return __tsc(u8"The protocol driver %1 is invalid.");
	case 0xc00a0003u:
		return __tsc(u8"The protocol driver %1 was not found in the system path.");
	case 0xc00a0006u:
		return __tsc(u8"A close operation is pending on the terminal connection.");
	case 0xc00a0007u:
		return __tsc(u8"No free output buffers are available.");
	case 0xc00a0008u:
		return __tsc(u8"The MODEM.INF file was not found.");
	case 0xc00a0009u:
		return __tsc(u8"The modem (%1) was not found in the MODEM.INF file.");
	case 0xc00a000au:
		return __tsc(u8"The modem did not accept the command sent to it. Verify that the configured modem name matches the attached modem.");
	case 0xc00a000bu:
		return __tsc(u8"The modem did not respond to the command sent to it. Verify that the modem cable is properly attached and the modem is turned on.");
	case 0xc00a000cu:
		return __tsc(u8"Carrier detection has failed or the carrier has been dropped due to disconnection.");
	case 0xc00a000du:
		return __tsc(u8"A dial tone was not detected within the required time. Verify that the phone cable is properly attached and functional.");
	case 0xc00a000eu:
		return __tsc(u8"A busy signal was detected at a remote site on callback.");
	case 0xc00a000fu:
		return __tsc(u8"A voice was detected at a remote site on callback.");
	case 0xc00a0010u:
		return __tsc(u8"Transport driver error.");
	case 0xc00a0012u:
		return __tsc(u8"The client you are using is not licensed to use this system. Your logon request is denied.");
	case 0xc00a0013u:
		return __tsc(u8"The system has reached its licensed logon limit. Try again later.");
	case 0xc00a0014u:
		return __tsc(u8"The system license has expired. Your logon request is denied.");
	case 0xc00a0015u:
		return __tsc(u8"The specified session cannot be found.");
	case 0xc00a0016u:
		return __tsc(u8"The specified session name is already in use.");
	case 0xc00a0017u:
		return __tsc(u8"The requested operation cannot be completed because the terminal connection is currently processing a connect, disconnect, reset, or delete operation.");
	case 0xc00a0018u:
		return __tsc(u8"An attempt has been made to connect to a session whose video mode is not supported by the current client.");
	case 0xc00a0022u:
		return __tsc(u8"The application attempted to enable DOS graphics mode. DOS graphics mode is not supported.");
	case 0xc00a0024u:
		return __tsc(u8"The requested operation can be performed only on the system console. This is most often the result of a driver or system DLL requiring direct console access.");
	case 0xc00a0026u:
		return __tsc(u8"The client failed to respond to the server connect message.");
	case 0xc00a0027u:
		return __tsc(u8"Disconnecting the console session is not supported.");
	case 0xc00a0028u:
		return __tsc(u8"Reconnecting a disconnected session to the console is not supported.");
	case 0xc00a002au:
		return __tsc(u8"The request to control another session remotely was denied.");
	case 0xc00a002bu:
		return __tsc(u8"A process has requested access to a session, but has not been granted those access rights.");
	case 0xc00a002eu:
		return __tsc(u8"The terminal connection driver %1 is invalid.");
	case 0xc00a002fu:
		return __tsc(u8"The terminal connection driver %1 was not found in the system path.");
	case 0xc00a0030u:
		return __tsc(u8"The requested session cannot be controlled remotely. You cannot control your own session, a session that is trying to control your session, a session that has no user logged on, or other sessions from the console.");
	case 0xc00a0031u:
		return __tsc(u8"The requested session is not configured to allow remote control.");
	case 0xc00a0032u:
		return __tsc(u8"The RDP protocol component %2 detected an error in the protocol stream and has disconnected the client.");
	case 0xc00a0033u:
		return __tsc(u8"Your request to connect to this terminal server has been rejected. Your terminal server client license number has not been entered for this copy of the terminal client. Contact your system administrator for help in entering a valid, unique license number for this terminal server client. Click OK to continue.");
	case 0xc00a0034u:
		return __tsc(u8"Your request to connect to this terminal server has been rejected. Your terminal server client license number is currently being used by another user. Contact your system administrator to obtain a new copy of the terminal server client with a valid, unique license number. Click OK to continue.");
	case 0xc00a0035u:
		return __tsc(u8"The remote control of the console was terminated because the display mode was changed. Changing the display mode in a remote control session is not supported.");
	case 0xc00a0036u:
		return __tsc(u8"Remote control could not be terminated because the specified session is not currently being remotely controlled.");
	case 0xc00a0037u:
		return __tsc(u8"Your interactive logon privilege has been disabled. Contact your system administrator.");
	case 0xc00a0038u:
		return __tsc(u8"The terminal server security layer detected an error in the protocol stream and has disconnected the client.");
	case 0xc00a0039u:
		return __tsc(u8"The target session is incompatible with the current session.");
	case 0xc00a003au:
		return __tsc(u8"");
	case 0xc00b0001u:
		return __tsc(u8"The resource loader failed to find an MUI file.");
	case 0xc00b0002u:
		return __tsc(u8"The resource loader failed to load an MUI file because the file failed to pass validation.");
	case 0xc00b0003u:
		return __tsc(u8"The RC manifest is corrupted with garbage data, is an unsupported version, or is missing a required item.");
	case 0xc00b0004u:
		return __tsc(u8"The RC manifest has an invalid culture name.");
	case 0xc00b0005u:
		return __tsc(u8"The RC manifest has and invalid ultimate fallback name.");
	case 0xc00b0006u:
		return __tsc(u8"The resource loader cache does not have a loaded MUI entry.");
	case 0xc00b0007u:
		return __tsc(u8"The user stopped resource enumeration.");
	case 0xc0130001u:
		return __tsc(u8"The cluster node is not valid.");
	case 0xc0130002u:
		return __tsc(u8"The cluster node already exists.");
	case 0xc0130003u:
		return __tsc(u8"A node is in the process of joining the cluster.");
	case 0xc0130004u:
		return __tsc(u8"The cluster node was not found.");
	case 0xc0130005u:
		return __tsc(u8"The cluster local node information was not found.");
	case 0xc0130006u:
		return __tsc(u8"The cluster network already exists.");
	case 0xc0130007u:
		return __tsc(u8"The cluster network was not found.");
	case 0xc0130008u:
		return __tsc(u8"The cluster network interface already exists.");
	case 0xc0130009u:
		return __tsc(u8"The cluster network interface was not found.");
	case 0xc013000au:
		return __tsc(u8"The cluster request is not valid for this object.");
	case 0xc013000bu:
		return __tsc(u8"The cluster network provider is not valid.");
	case 0xc013000cu:
		return __tsc(u8"The cluster node is down.");
	case 0xc013000du:
		return __tsc(u8"The cluster node is not reachable.");
	case 0xc013000eu:
		return __tsc(u8"The cluster node is not a member of the cluster.");
	case 0xc013000fu:
		return __tsc(u8"A cluster join operation is not in progress.");
	case 0xc0130010u:
		return __tsc(u8"The cluster network is not valid.");
	case 0xc0130011u:
		return __tsc(u8"No network adapters are available.");
	case 0xc0130012u:
		return __tsc(u8"The cluster node is up.");
	case 0xc0130013u:
		return __tsc(u8"The cluster node is paused.");
	case 0xc0130014u:
		return __tsc(u8"The cluster node is not paused.");
	case 0xc0130015u:
		return __tsc(u8"No cluster security context is available.");
	case 0xc0130016u:
		return __tsc(u8"The cluster network is not configured for internal cluster communication.");
	case 0xc0130017u:
		return __tsc(u8"The cluster node has been poisoned.");
	case 0xc0130018u:
	case 0xc0130019u:
	case 0xc0130020u:
	case 0xc0130021u:
	case 0xc0130022u:
	case 0xc0130023u:
	case 0xc0130024u:
	case 0xc0130025u:
	case 0xc0130026u:
	case 0xc0130027u:
	case 0xc0130028u:
	case 0xc0130029u:
	case 0xc0130030u:
	case 0xc0130031u:
		return __tsc(u8"");
	case 0xc0140001u:
		return __tsc(u8"An attempt was made to run an invalid AML opcode.");
	case 0xc0140002u:
		return __tsc(u8"The AML interpreter stack has overflowed.");
	case 0xc0140003u:
		return __tsc(u8"An inconsistent state has occurred.");
	case 0xc0140004u:
		return __tsc(u8"An attempt was made to access an array outside its bounds.");
	case 0xc0140005u:
		return __tsc(u8"A required argument was not specified.");
	case 0xc0140006u:
		return __tsc(u8"A fatal error has occurred.");
	case 0xc0140007u:
		return __tsc(u8"An invalid SuperName was specified.");
	case 0xc0140008u:
		return __tsc(u8"An argument with an incorrect type was specified.");
	case 0xc0140009u:
		return __tsc(u8"An object with an incorrect type was specified.");
	case 0xc014000au:
		return __tsc(u8"A target with an incorrect type was specified.");
	case 0xc014000bu:
		return __tsc(u8"An incorrect number of arguments was specified.");
	case 0xc014000cu:
		return __tsc(u8"An address failed to translate.");
	case 0xc014000du:
		return __tsc(u8"An incorrect event type was specified.");
	case 0xc014000eu:
		return __tsc(u8"A handler for the target already exists.");
	case 0xc014000fu:
		return __tsc(u8"Invalid data for the target was specified.");
	case 0xc0140010u:
		return __tsc(u8"An invalid region for the target was specified.");
	case 0xc0140011u:
		return __tsc(u8"An attempt was made to access a field outside the defined range.");
	case 0xc0140012u:
		return __tsc(u8"The global system lock could not be acquired.");
	case 0xc0140013u:
		return __tsc(u8"An attempt was made to reinitialize the ACPI subsystem.");
	case 0xc0140014u:
		return __tsc(u8"The ACPI subsystem has not been initialized.");
	case 0xc0140015u:
		return __tsc(u8"An incorrect mutex was specified.");
	case 0xc0140016u:
		return __tsc(u8"The mutex is not currently owned.");
	case 0xc0140017u:
		return __tsc(u8"An attempt was made to access the mutex by a process that was not the owner.");
	case 0xc0140018u:
		return __tsc(u8"An error occurred during an access to region space.");
	case 0xc0140019u:
		return __tsc(u8"An attempt was made to use an incorrect table.");
	case 0xc0140020u:
		return __tsc(u8"The registration of an ACPI event failed.");
	case 0xc0140021u:
		return __tsc(u8"An ACPI power object failed to transition state.");
	case 0xc0150001u:
		return __tsc(u8"The requested section is not present in the activation context.");
	case 0xc0150002u:
		return __tsc(u8"Windows was unble to process the application binding information. Refer to the system event log for further information.");
	case 0xc0150003u:
		return __tsc(u8"The application binding data format is invalid.");
	case 0xc0150004u:
		return __tsc(u8"The referenced assembly is not installed on the system.");
	case 0xc0150005u:
		return __tsc(u8"The manifest file does not begin with the required tag and format information.");
	case 0xc0150006u:
		return __tsc(u8"The manifest file contains one or more syntax errors.");
	case 0xc0150007u:
		return __tsc(u8"The application attempted to activate a disabled activation context.");
	case 0xc0150008u:
		return __tsc(u8"The requested lookup key was not found in any active activation context.");
	case 0xc0150009u:
		return __tsc(u8"A component version required by the application conflicts with another component version that is already active.");
	case 0xc015000au:
		return __tsc(u8"The type requested activation context section does not match the query API used.");
	case 0xc015000bu:
		return __tsc(u8"Lack of system resources has required isolated activation to be disabled for the current thread of execution.");
	case 0xc015000cu:
		return __tsc(u8"The referenced assembly could not be found.");
	case 0xc015000eu:
		return __tsc(u8"An attempt to set the process default activation context failed because the process default activation context was already set.");
	case 0xc015000fu:
		return __tsc(u8"The activation context being deactivated is not the most recently activated one.");
	case 0xc0150010u:
		return __tsc(u8"The activation context being deactivated is not active for the current thread of execution.");
	case 0xc0150011u:
		return __tsc(u8"The activation context being deactivated has already been deactivated.");
	case 0xc0150012u:
		return __tsc(u8"The activation context of the system default assembly could not be generated.");
	case 0xc0150013u:
		return __tsc(u8"A component used by the isolation facility has requested that the process be terminated.");
	case 0xc0150014u:
		return __tsc(u8"The activation context activation stack for the running thread of execution is corrupt.");
	case 0xc0150015u:
		return __tsc(u8"The application isolation metadata for this process or thread has become corrupt.");
	case 0xc0150016u:
		return __tsc(u8"The value of an attribute in an identity is not within the legal range.");
	case 0xc0150017u:
		return __tsc(u8"The name of an attribute in an identity is not within the legal range.");
	case 0xc0150018u:
		return __tsc(u8"An identity contains two definitions for the same attribute.");
	case 0xc0150019u:
		return __tsc(u8"The identity string is malformed. This might be due to a trailing comma, more than two unnamed attributes, a missing attribute name, or a missing attribute value.");
	case 0xc015001au:
		return __tsc(u8"The component store has become corrupted.");
	case 0xc015001bu:
		return __tsc(u8"A component's file does not match the verification information present in the component manifest.");
	case 0xc015001cu:
		return __tsc(u8"The identities of the manifests are identical, but their contents are different.");
	case 0xc015001du:
		return __tsc(u8"The component identities are different.");
	case 0xc015001eu:
		return __tsc(u8"The assembly is not a deployment.");
	case 0xc015001fu:
		return __tsc(u8"The file is not a part of the assembly.");
	case 0xc0150020u:
		return __tsc(u8"An advanced installer failed during setup or servicing.");
	case 0xc0150021u:
		return __tsc(u8"The character encoding in the XML declaration did not match the encoding used in the document.");
	case 0xc0150022u:
		return __tsc(u8"The size of the manifest exceeds the maximum allowed.");
	case 0xc0150023u:
		return __tsc(u8"The setting is not registered.");
	case 0xc0150024u:
		return __tsc(u8"One or more required transaction members are not present.");
	case 0xc0150025u:
		return __tsc(u8"The SMI primitive installer failed during setup or servicing.");
	case 0xc0150026u:
		return __tsc(u8"A generic command executable returned a result that indicates failure.");
	case 0xc0150027u:
		return __tsc(u8"A component is missing file verification information in its manifest.");
	case 0xc0190001u:
		return __tsc(u8"The function attempted to use a name that is reserved for use by another transaction.");
	case 0xc0190002u:
		return __tsc(u8"The transaction handle associated with this operation is invalid.");
	case 0xc0190003u:
		return __tsc(u8"The requested operation was made in the context of a transaction that is no longer active.");
	case 0xc0190004u:
		return __tsc(u8"The transaction manager was unable to be successfully initialized. Transacted operations are not supported.");
	case 0xc0190005u:
		return __tsc(u8"Transaction support within the specified file system resource manager was not started or was shut down due to an error.");
	case 0xc0190006u:
		return __tsc(u8"The metadata of the resource manager has been corrupted. The resource manager will not function.");
	case 0xc0190007u:
		return __tsc(u8"The resource manager attempted to prepare a transaction that it has not successfully joined.");
	case 0xc0190008u:
		return __tsc(u8"The specified directory does not contain a file system resource manager.");
	case 0xc019000au:
		return __tsc(u8"The remote server or share does not support transacted file operations.");
	case 0xc019000bu:
		return __tsc(u8"The requested log size for the file system resource manager is invalid.");
	case 0xc019000cu:
		return __tsc(u8"The remote server sent mismatching version number or Fid for a file opened with transactions.");
	case 0xc019000fu:
		return __tsc(u8"The resource manager tried to register a protocol that already exists.");
	case 0xc0190010u:
		return __tsc(u8"The attempt to propagate the transaction failed.");
	case 0xc0190011u:
		return __tsc(u8"The requested propagation protocol was not registered as a CRM.");
	case 0xc0190012u:
		return __tsc(u8"The transaction object already has a superior enlistment, and the caller attempted an operation that would have created a new superior. Only a single superior enlistment is allowed.");
	case 0xc0190013u:
		return __tsc(u8"The requested operation is not valid on the transaction object in its current state.");
	case 0xc0190014u:
		return __tsc(u8"The caller has called a response API, but the response is not expected because the transaction manager did not issue the corresponding request to the caller.");
	case 0xc0190015u:
		return __tsc(u8"It is too late to perform the requested operation, because the transaction has already been aborted.");
	case 0xc0190016u:
		return __tsc(u8"It is too late to perform the requested operation, because the transaction has already been committed.");
	case 0xc0190017u:
		return __tsc(u8"The buffer passed in to NtPushTransaction or NtPullTransaction is not in a valid format.");
	case 0xc0190018u:
		return __tsc(u8"The current transaction context associated with the thread is not a valid handle to a transaction object.");
	case 0xc0190019u:
		return __tsc(u8"An attempt to create space in the transactional resource manager's log failed. The failure status has been recorded in the event log.");
	case 0xc0190021u:
		return __tsc(u8"The object (file, stream, or link) that corresponds to the handle has been deleted by a transaction savepoint rollback.");
	case 0xc0190022u:
		return __tsc(u8"The specified file miniversion was not found for this transacted file open.");
	case 0xc0190023u:
		return __tsc(u8"The specified file miniversion was found but has been invalidated. The most likely cause is a transaction savepoint rollback.");
	case 0xc0190024u:
		return __tsc(u8"A miniversion can be opened only in the context of the transaction that created it.");
	case 0xc0190025u:
		return __tsc(u8"It is not possible to open a miniversion with modify access.");
	case 0xc0190026u:
		return __tsc(u8"It is not possible to create any more miniversions for this stream.");
	case 0xc0190028u:
		return __tsc(u8"The handle has been invalidated by a transaction. The most likely cause is the presence of memory mapping on a file or an open handle when the transaction ended or rolled back to savepoint.");
	case 0xc0190030u:
		return __tsc(u8"The log data is corrupt.");
	case 0xc0190032u:
		return __tsc(u8"The transaction outcome is unavailable because the resource manager responsible for it is disconnected.");
	case 0xc0190033u:
		return __tsc(u8"The request was rejected because the enlistment in question is not a superior enlistment.");
	case 0xc0190036u:
		return __tsc(u8"The file cannot be opened in a transaction because its identity depends on the outcome of an unresolved transaction.");
	case 0xc0190037u:
		return __tsc(u8"The operation cannot be performed because another transaction is depending on this property not changing.");
	case 0xc0190038u:
		return __tsc(u8"The operation would involve a single file with two transactional resource managers and is, therefore, not allowed.");
	case 0xc0190039u:
		return __tsc(u8"The $Txf directory must be empty for this operation to succeed.");
	case 0xc019003au:
		return __tsc(u8"The operation would leave a transactional resource manager in an inconsistent state and is therefore not allowed.");
	case 0xc019003bu:
		return __tsc(u8"The operation could not be completed because the transaction manager does not have a log.");
	case 0xc019003cu:
		return __tsc(u8"A rollback could not be scheduled because a previously scheduled rollback has already executed or been queued for execution.");
	case 0xc019003du:
		return __tsc(u8"The transactional metadata attribute on the file or directory %hs is corrupt and unreadable.");
	case 0xc019003eu:
		return __tsc(u8"The encryption operation could not be completed because a transaction is active.");
	case 0xc019003fu:
		return __tsc(u8"This object is not allowed to be opened in a transaction.");
	case 0xc0190040u:
		return __tsc(u8"Memory mapping (creating a mapped section) a remote file under a transaction is not supported.");
	case 0xc0190043u:
		return __tsc(u8"Promotion was required to allow the resource manager to enlist, but the transaction was set to disallow it.");
	case 0xc0190044u:
		return __tsc(u8"This file is open for modification in an unresolved transaction and can be opened for execute only by a transacted reader.");
	case 0xc0190045u:
		return __tsc(u8"The request to thaw frozen transactions was ignored because transactions were not previously frozen.");
	case 0xc0190046u:
		return __tsc(u8"Transactions cannot be frozen because a freeze is already in progress.");
	case 0xc0190047u:
		return __tsc(u8"The target volume is not a snapshot volume. This operation is valid only on a volume mounted as a snapshot.");
	case 0xc0190048u:
		return __tsc(u8"The savepoint operation failed because files are open on the transaction, which is not permitted.");
	case 0xc0190049u:
		return __tsc(u8"The sparse operation could not be completed because a transaction is active on the file.");
	case 0xc019004au:
		return __tsc(u8"The call to create a transaction manager object failed because the Tm Identity that is stored in the log file does not match the Tm Identity that was passed in as an argument.");
	case 0xc019004bu:
		return __tsc(u8"I/O was attempted on a section object that has been floated as a result of a transaction ending. There is no valid data.");
	case 0xc019004cu:
		return __tsc(u8"The transactional resource manager cannot currently accept transacted work due to a transient condition, such as low resources.");
	case 0xc019004du:
		return __tsc(u8"The transactional resource manager had too many transactions outstanding that could not be aborted. The transactional resource manager has been shut down.");
	case 0xc019004eu:
		return __tsc(u8"The specified transaction was unable to be opened because it was not found.");
	case 0xc019004fu:
		return __tsc(u8"The specified resource manager was unable to be opened because it was not found.");
	case 0xc0190050u:
		return __tsc(u8"The specified enlistment was unable to be opened because it was not found.");
	case 0xc0190051u:
		return __tsc(u8"The specified transaction manager was unable to be opened because it was not found.");
	case 0xc0190052u:
		return __tsc(u8"The specified resource manager was unable to create an enlistment because its associated transaction manager is not online.");
	case 0xc0190053u:
		return __tsc(u8"The specified transaction manager was unable to create the objects contained in its log file in the Ob namespace. Therefore, the transaction manager was unable to recover.");
	case 0xc0190054u:
		return __tsc(u8"The call to create a superior enlistment on this transaction object could not be completed because the transaction object specified for the enlistment is a subordinate branch of the transaction. Only the root of the transaction can be enlisted as a superior.");
	case 0xc0190055u:
		return __tsc(u8"Because the associated transaction manager or resource manager has been closed, the handle is no longer valid.");
	case 0xc0190056u:
		return __tsc(u8"The compression operation could not be completed because a transaction is active on the file.");
	case 0xc0190057u:
		return __tsc(u8"The specified operation could not be performed on this superior enlistment because the enlistment was not created with the corresponding completion response in the NotificationMask.");
	case 0xc0190058u:
		return __tsc(u8"The specified operation could not be performed because the record to be logged was too long. This can occur because either there are too many enlistments on this transaction or the combined RecoveryInformation being logged on behalf of those enlistments is too long.");
	case 0xc0190059u:
		return __tsc(u8"The link-tracking operation could not be completed because a transaction is active.");
	case 0xc019005au:
		return __tsc(u8"This operation cannot be performed in a transaction.");
	case 0xc019005bu:
		return __tsc(u8"The kernel transaction manager had to abort or forget the transaction because it blocked forward progress.");
	case 0xc019005cu:
	case 0xc019005du:
	case 0xc019005eu:
	case 0xc019005fu:
		return __tsc(u8"");
	case 0xc0190060u:
		return __tsc(u8"The handle is no longer properly associated with its transaction. It might have been opened in a transactional resource manager that was subsequently forced to restart. Please close the handle and open a new one.");
	case 0xc0190061u:
		return __tsc(u8"The specified operation could not be performed because the resource manager is not enlisted in the transaction.");
	case 0xc01a0001u:
		return __tsc(u8"The log service found an invalid log sector.");
	case 0xc01a0002u:
		return __tsc(u8"The log service encountered a log sector with invalid block parity.");
	case 0xc01a0003u:
		return __tsc(u8"The log service encountered a remapped log sector.");
	case 0xc01a0004u:
		return __tsc(u8"The log service encountered a partial or incomplete log block.");
	case 0xc01a0005u:
		return __tsc(u8"The log service encountered an attempt to access data outside the active log range.");
	case 0xc01a0006u:
		return __tsc(u8"The log service user-log marshaling buffers are exhausted.");
	case 0xc01a0007u:
		return __tsc(u8"The log service encountered an attempt to read from a marshaling area with an invalid read context.");
	case 0xc01a0008u:
		return __tsc(u8"The log service encountered an invalid log restart area.");
	case 0xc01a0009u:
		return __tsc(u8"The log service encountered an invalid log block version.");
	case 0xc01a000au:
		return __tsc(u8"The log service encountered an invalid log block.");
	case 0xc01a000bu:
		return __tsc(u8"The log service encountered an attempt to read the log with an invalid read mode.");
	case 0xc01a000du:
		return __tsc(u8"The log service encountered a corrupted metadata file.");
	case 0xc01a000eu:
		return __tsc(u8"The log service encountered a metadata file that could not be created by the log file system.");
	case 0xc01a000fu:
		return __tsc(u8"The log service encountered a metadata file with inconsistent data.");
	case 0xc01a0010u:
		return __tsc(u8"The log service encountered an attempt to erroneously allocate or dispose reservation space.");
	case 0xc01a0011u:
		return __tsc(u8"The log service cannot delete the log file or the file system container.");
	case 0xc01a0012u:
		return __tsc(u8"The log service has reached the maximum allowable containers allocated to a log file.");
	case 0xc01a0013u:
		return __tsc(u8"The log service has attempted to read or write backward past the start of the log.");
	case 0xc01a0014u:
		return __tsc(u8"The log policy could not be installed because a policy of the same type is already present.");
	case 0xc01a0015u:
		return __tsc(u8"The log policy in question was not installed at the time of the request.");
	case 0xc01a0016u:
		return __tsc(u8"The installed set of policies on the log is invalid.");
	case 0xc01a0017u:
		return __tsc(u8"A policy on the log in question prevented the operation from completing.");
	case 0xc01a0018u:
		return __tsc(u8"The log space cannot be reclaimed because the log is pinned by the archive tail.");
	case 0xc01a0019u:
		return __tsc(u8"The log record is not a record in the log file.");
	case 0xc01a001au:
		return __tsc(u8"The number of reserved log records or the adjustment of the number of reserved log records is invalid.");
	case 0xc01a001bu:
		return __tsc(u8"The reserved log space or the adjustment of the log space is invalid.");
	case 0xc01a001cu:
		return __tsc(u8"A new or existing archive tail or the base of the active log is invalid.");
	case 0xc01a001du:
		return __tsc(u8"The log space is exhausted.");
	case 0xc01a001eu:
		return __tsc(u8"The log is multiplexed; no direct writes to the physical log are allowed.");
	case 0xc01a001fu:
		return __tsc(u8"The operation failed because the log is dedicated.");
	case 0xc01a0020u:
		return __tsc(u8"The operation requires an archive context.");
	case 0xc01a0021u:
		return __tsc(u8"Log archival is in progress.");
	case 0xc01a0022u:
		return __tsc(u8"The operation requires a nonephemeral log, but the log is ephemeral.");
	case 0xc01a0023u:
		return __tsc(u8"The log must have at least two containers before it can be read from or written to.");
	case 0xc01a0024u:
		return __tsc(u8"A log client has already registered on the stream.");
	case 0xc01a0025u:
		return __tsc(u8"A log client has not been registered on the stream.");
	case 0xc01a0026u:
		return __tsc(u8"A request has already been made to handle the log full condition.");
	case 0xc01a0027u:
		return __tsc(u8"The log service encountered an error when attempting to read from a log container.");
	case 0xc01a0028u:
		return __tsc(u8"The log service encountered an error when attempting to write to a log container.");
	case 0xc01a0029u:
		return __tsc(u8"The log service encountered an error when attempting to open a log container.");
	case 0xc01a002au:
		return __tsc(u8"The log service encountered an invalid container state when attempting a requested action.");
	case 0xc01a002bu:
		return __tsc(u8"The log service is not in the correct state to perform a requested action.");
	case 0xc01a002cu:
		return __tsc(u8"The log space cannot be reclaimed because the log is pinned.");
	case 0xc01a002du:
		return __tsc(u8"The log metadata flush failed.");
	case 0xc01a002eu:
		return __tsc(u8"Security on the log and its containers is inconsistent.");
	case 0xc01a002fu:
		return __tsc(u8"Records were appended to the log or reservation changes were made, but the log could not be flushed.");
	case 0xc01a0030u:
		return __tsc(u8"The log is pinned due to reservation consuming most of the log space. Free some reserved records to make space available.");
	case 0xc01b00eau:
		return __tsc(u8"{Display Driver Stopped Responding} The %hs display driver has stopped working normally. Save your work and reboot the system to restore full display functionality. The next time you reboot the computer, a dialog box will allow you to upload data about this failure to Microsoft.");
	case 0xc01c0001u:
		return __tsc(u8"A handler was not defined by the filter for this operation.");
	case 0xc01c0002u:
		return __tsc(u8"A context is already defined for this object.");
	case 0xc01c0003u:
		return __tsc(u8"Asynchronous requests are not valid for this operation.");
	case 0xc01c0004u:
		return __tsc(u8"This is an internal error code used by the filter manager to determine if a fast I/O operation should be forced down the input/output request packet (IRP) path. Minifilters should never return this value.");
	case 0xc01c0005u:
		return __tsc(u8"An invalid name request was made. The name requested cannot be retrieved at this time.");
	case 0xc01c0006u:
		return __tsc(u8"Posting this operation to a worker thread for further processing is not safe at this time because it could lead to a system deadlock.");
	case 0xc01c0007u:
		return __tsc(u8"The Filter Manager was not initialized when a filter tried to register. Make sure that the Filter Manager is loaded as a driver.");
	case 0xc01c0008u:
		return __tsc(u8"The filter is not ready for attachment to volumes because it has not finished initializing (FltStartFiltering has not been called).");
	case 0xc01c0009u:
		return __tsc(u8"The filter must clean up any operation-specific context at this time because it is being removed from the system before the operation is completed by the lower drivers.");
	case 0xc01c000au:
		return __tsc(u8"The Filter Manager had an internal error from which it cannot recover; therefore, the operation has failed. This is usually the result of a filter returning an invalid value from a pre-operation callback.");
	case 0xc01c000bu:
		return __tsc(u8"The object specified for this action is in the process of being deleted; therefore, the action requested cannot be completed at this time.");
	case 0xc01c000cu:
		return __tsc(u8"A nonpaged pool must be used for this type of context.");
	case 0xc01c000du:
		return __tsc(u8"A duplicate handler definition has been provided for an operation.");
	case 0xc01c000eu:
		return __tsc(u8"The callback data queue has been disabled.");
	case 0xc01c000fu:
		return __tsc(u8"Do not attach the filter to the volume at this time.");
	case 0xc01c0010u:
		return __tsc(u8"Do not detach the filter from the volume at this time.");
	case 0xc01c0011u:
		return __tsc(u8"An instance already exists at this altitude on the volume specified.");
	case 0xc01c0012u:
		return __tsc(u8"An instance already exists with this name on the volume specified.");
	case 0xc01c0013u:
		return __tsc(u8"The system could not find the filter specified.");
	case 0xc01c0014u:
		return __tsc(u8"The system could not find the volume specified.");
	case 0xc01c0015u:
		return __tsc(u8"The system could not find the instance specified.");
	case 0xc01c0016u:
		return __tsc(u8"No registered context allocation definition was found for the given request.");
	case 0xc01c0017u:
		return __tsc(u8"An invalid parameter was specified during context registration.");
	case 0xc01c0018u:
		return __tsc(u8"The name requested was not found in the Filter Manager name cache and could not be retrieved from the file system.");
	case 0xc01c0019u:
		return __tsc(u8"The requested device object does not exist for the given volume.");
	case 0xc01c001au:
		return __tsc(u8"The specified volume is already mounted.");
	case 0xc01c001bu:
		return __tsc(u8"The specified transaction context is already enlisted in a transaction.");
	case 0xc01c001cu:
		return __tsc(u8"The specified context is already attached to another object.");
	case 0xc01c0020u:
		return __tsc(u8"No waiter is present for the filter's reply to this message.");
	case 0xc01c0023u:
	case 0xc01c0024u:
		return __tsc(u8"");
	case 0xc01d0001u:
		return __tsc(u8"A monitor descriptor could not be obtained.");
	case 0xc01d0002u:
		return __tsc(u8"This release does not support the format of the obtained monitor descriptor.");
	case 0xc01d0003u:
		return __tsc(u8"The checksum of the obtained monitor descriptor is invalid.");
	case 0xc01d0004u:
		return __tsc(u8"The monitor descriptor contains an invalid standard timing block.");
	case 0xc01d0005u:
		return __tsc(u8"WMI data-block registration failed for one of the MSMonitorClass WMI subclasses.");
	case 0xc01d0006u:
		return __tsc(u8"The provided monitor descriptor block is either corrupted or does not contain the monitor's detailed serial number.");
	case 0xc01d0007u:
		return __tsc(u8"The provided monitor descriptor block is either corrupted or does not contain the monitor's user-friendly name.");
	case 0xc01d0008u:
		return __tsc(u8"There is no monitor descriptor data at the specified (offset or size) region.");
	case 0xc01d0009u:
		return __tsc(u8"The monitor descriptor contains an invalid detailed timing block.");
	case 0xc01d000au:
		return __tsc(u8"Monitor descriptor contains invalid manufacture date.");
	case 0xc01e0000u:
		return __tsc(u8"Exclusive mode ownership is needed to create an unmanaged primary allocation.");
	case 0xc01e0001u:
		return __tsc(u8"The driver needs more DMA buffer space to complete the requested operation.");
	case 0xc01e0002u:
		return __tsc(u8"The specified display adapter handle is invalid.");
	case 0xc01e0003u:
		return __tsc(u8"The specified display adapter and all of its state have been reset.");
	case 0xc01e0004u:
		return __tsc(u8"The driver stack does not match the expected driver model.");
	case 0xc01e0005u:
		return __tsc(u8"Present happened but ended up into the changed desktop mode.");
	case 0xc01e0006u:
		return __tsc(u8"Nothing to present due to desktop occlusion.");
	case 0xc01e0007u:
		return __tsc(u8"Not able to present due to denial of desktop access.");
	case 0xc01e0008u:
		return __tsc(u8"Not able to present with color conversion.");
	case 0xc01e0009u:
		return __tsc(u8"");
	case 0xc01e000bu:
		return __tsc(u8"Present redirection is disabled (desktop windowing management subsystem is off).");
	case 0xc01e000cu:
		return __tsc(u8"Previous exclusive VidPn source owner has released its ownership");
	case 0xc01e000du:
	case 0xc01e000eu:
	case 0xc01e000fu:
	case 0xc01e0010u:
	case 0xc01e0011u:
	case 0xc01e0012u:
	case 0xc01e0013u:
	case 0xc01e0018u:
		return __tsc(u8"");
	case 0xc01e0100u:
		return __tsc(u8"Not enough video memory is available to complete the operation.");
	case 0xc01e0101u:
		return __tsc(u8"Could not probe and lock the underlying memory of an allocation.");
	case 0xc01e0102u:
		return __tsc(u8"The allocation is currently busy.");
	case 0xc01e0103u:
		return __tsc(u8"An object being referenced has already reached the maximum reference count and cannot be referenced further.");
	case 0xc01e0104u:
		return __tsc(u8"A problem could not be solved due to an existing condition. Try again later.");
	case 0xc01e0105u:
		return __tsc(u8"A problem could not be solved due to an existing condition. Try again now.");
	case 0xc01e0106u:
		return __tsc(u8"The allocation is invalid.");
	case 0xc01e0107u:
		return __tsc(u8"No more unswizzling apertures are currently available.");
	case 0xc01e0108u:
		return __tsc(u8"The current allocation cannot be unswizzled by an aperture.");
	case 0xc01e0109u:
		return __tsc(u8"The request failed because a pinned allocation cannot be evicted.");
	case 0xc01e0110u:
		return __tsc(u8"The allocation cannot be used from its current segment location for the specified operation.");
	case 0xc01e0111u:
		return __tsc(u8"A locked allocation cannot be used in the current command buffer.");
	case 0xc01e0112u:
		return __tsc(u8"The allocation being referenced has been closed permanently.");
	case 0xc01e0113u:
		return __tsc(u8"An invalid allocation instance is being referenced.");
	case 0xc01e0114u:
		return __tsc(u8"An invalid allocation handle is being referenced.");
	case 0xc01e0115u:
		return __tsc(u8"The allocation being referenced does not belong to the current device.");
	case 0xc01e0116u:
		return __tsc(u8"The specified allocation lost its content.");
	case 0xc01e0200u:
		return __tsc(u8"A GPU exception was detected on the given device. The device cannot be scheduled.");
	case 0xc01e0300u:
		return __tsc(u8"The specified VidPN topology is invalid.");
	case 0xc01e0301u:
		return __tsc(u8"The specified VidPN topology is valid but is not supported by this model of the display adapter.");
	case 0xc01e0302u:
		return __tsc(u8"The specified VidPN topology is valid but is not currently supported by the display adapter due to allocation of its resources.");
	case 0xc01e0303u:
		return __tsc(u8"The specified VidPN handle is invalid.");
	case 0xc01e0304u:
		return __tsc(u8"The specified video present source is invalid.");
	case 0xc01e0305u:
		return __tsc(u8"The specified video present target is invalid.");
	case 0xc01e0306u:
		return __tsc(u8"The specified VidPN modality is not supported (for example, at least two of the pinned modes are not co-functional).");
	case 0xc01e0308u:
		return __tsc(u8"The specified VidPN source mode set is invalid.");
	case 0xc01e0309u:
		return __tsc(u8"The specified VidPN target mode set is invalid.");
	case 0xc01e030au:
		return __tsc(u8"The specified video signal frequency is invalid.");
	case 0xc01e030bu:
		return __tsc(u8"The specified video signal active region is invalid.");
	case 0xc01e030cu:
		return __tsc(u8"The specified video signal total region is invalid.");
	case 0xc01e0310u:
		return __tsc(u8"The specified video present source mode is invalid.");
	case 0xc01e0311u:
		return __tsc(u8"The specified video present target mode is invalid.");
	case 0xc01e0312u:
		return __tsc(u8"The pinned mode must remain in the set on the VidPN's co-functional modality enumeration.");
	case 0xc01e0313u:
		return __tsc(u8"The specified video present path is already in the VidPN's topology.");
	case 0xc01e0314u:
		return __tsc(u8"The specified mode is already in the mode set.");
	case 0xc01e0315u:
		return __tsc(u8"The specified video present source set is invalid.");
	case 0xc01e0316u:
		return __tsc(u8"The specified video present target set is invalid.");
	case 0xc01e0317u:
		return __tsc(u8"The specified video present source is already in the video present source set.");
	case 0xc01e0318u:
		return __tsc(u8"The specified video present target is already in the video present target set.");
	case 0xc01e0319u:
		return __tsc(u8"The specified VidPN present path is invalid.");
	case 0xc01e031au:
		return __tsc(u8"The miniport has no recommendation for augmenting the specified VidPN's topology.");
	case 0xc01e031bu:
		return __tsc(u8"The specified monitor frequency range set is invalid.");
	case 0xc01e031cu:
		return __tsc(u8"The specified monitor frequency range is invalid.");
	case 0xc01e031du:
		return __tsc(u8"The specified frequency range is not in the specified monitor frequency range set.");
	case 0xc01e031fu:
		return __tsc(u8"The specified frequency range is already in the specified monitor frequency range set.");
	case 0xc01e0320u:
		return __tsc(u8"The specified mode set is stale. Reacquire the new mode set.");
	case 0xc01e0321u:
		return __tsc(u8"The specified monitor source mode set is invalid.");
	case 0xc01e0322u:
		return __tsc(u8"The specified monitor source mode is invalid.");
	case 0xc01e0323u:
		return __tsc(u8"The miniport does not have a recommendation regarding the request to provide a functional VidPN given the current display adapter configuration.");
	case 0xc01e0324u:
		return __tsc(u8"The ID of the specified mode is being used by another mode in the set.");
	case 0xc01e0325u:
		return __tsc(u8"The system failed to determine a mode that is supported by both the display adapter and the monitor connected to it.");
	case 0xc01e0326u:
		return __tsc(u8"The number of video present targets must be greater than or equal to the number of video present sources.");
	case 0xc01e0327u:
		return __tsc(u8"The specified present path is not in the VidPN's topology.");
	case 0xc01e0328u:
		return __tsc(u8"The display adapter must have at least one video present source.");
	case 0xc01e0329u:
		return __tsc(u8"The display adapter must have at least one video present target.");
	case 0xc01e032au:
		return __tsc(u8"The specified monitor descriptor set is invalid.");
	case 0xc01e032bu:
		return __tsc(u8"The specified monitor descriptor is invalid.");
	case 0xc01e032cu:
		return __tsc(u8"The specified descriptor is not in the specified monitor descriptor set.");
	case 0xc01e032du:
		return __tsc(u8"The specified descriptor is already in the specified monitor descriptor set.");
	case 0xc01e032eu:
		return __tsc(u8"The ID of the specified monitor descriptor is being used by another descriptor in the set.");
	case 0xc01e032fu:
		return __tsc(u8"The specified video present target subset type is invalid.");
	case 0xc01e0330u:
		return __tsc(u8"Two or more of the specified resources are not related to each other, as defined by the interface semantics.");
	case 0xc01e0331u:
		return __tsc(u8"The ID of the specified video present source is being used by another source in the set.");
	case 0xc01e0332u:
		return __tsc(u8"The ID of the specified video present target is being used by another target in the set.");
	case 0xc01e0333u:
		return __tsc(u8"The specified VidPN source cannot be used because there is no available VidPN target to connect it to.");
	case 0xc01e0334u:
		return __tsc(u8"The newly arrived monitor could not be associated with a display adapter.");
	case 0xc01e0335u:
		return __tsc(u8"The particular display adapter does not have an associated VidPN manager.");
	case 0xc01e0336u:
		return __tsc(u8"The VidPN manager of the particular display adapter does not have an active VidPN.");
	case 0xc01e0337u:
		return __tsc(u8"The specified VidPN topology is stale; obtain the new topology.");
	case 0xc01e0338u:
		return __tsc(u8"No monitor is connected on the specified video present target.");
	case 0xc01e0339u:
		return __tsc(u8"The specified source is not part of the specified VidPN's topology.");
	case 0xc01e033au:
		return __tsc(u8"The specified primary surface size is invalid.");
	case 0xc01e033bu:
		return __tsc(u8"The specified visible region size is invalid.");
	case 0xc01e033cu:
		return __tsc(u8"The specified stride is invalid.");
	case 0xc01e033du:
		return __tsc(u8"The specified pixel format is invalid.");
	case 0xc01e033eu:
		return __tsc(u8"The specified color basis is invalid.");
	case 0xc01e033fu:
		return __tsc(u8"The specified pixel value access mode is invalid.");
	case 0xc01e0340u:
		return __tsc(u8"The specified target is not part of the specified VidPN's topology.");
	case 0xc01e0341u:
		return __tsc(u8"Failed to acquire the display mode management interface.");
	case 0xc01e0342u:
		return __tsc(u8"The specified VidPN source is already owned by a DMM client and cannot be used until that client releases it.");
	case 0xc01e0343u:
		return __tsc(u8"The specified VidPN is active and cannot be accessed.");
	case 0xc01e0344u:
		return __tsc(u8"The specified VidPN's present path importance ordinal is invalid.");
	case 0xc01e0345u:
		return __tsc(u8"The specified VidPN's present path content geometry transformation is invalid.");
	case 0xc01e0346u:
		return __tsc(u8"The specified content geometry transformation is not supported on the respective VidPN present path.");
	case 0xc01e0347u:
		return __tsc(u8"The specified gamma ramp is invalid.");
	case 0xc01e0348u:
		return __tsc(u8"The specified gamma ramp is not supported on the respective VidPN present path.");
	case 0xc01e0349u:
		return __tsc(u8"Multisampling is not supported on the respective VidPN present path.");
	case 0xc01e034au:
		return __tsc(u8"The specified mode is not in the specified mode set.");
	case 0xc01e034du:
		return __tsc(u8"The specified VidPN topology recommendation reason is invalid.");
	case 0xc01e034eu:
		return __tsc(u8"The specified VidPN present path content type is invalid.");
	case 0xc01e034fu:
		return __tsc(u8"The specified VidPN present path copy protection type is invalid.");
	case 0xc01e0350u:
		return __tsc(u8"Only one unassigned mode set can exist at any one time for a particular VidPN source or target.");
	case 0xc01e0352u:
		return __tsc(u8"The specified scan line ordering type is invalid.");
	case 0xc01e0353u:
		return __tsc(u8"The topology changes are not allowed for the specified VidPN.");
	case 0xc01e0354u:
		return __tsc(u8"All available importance ordinals are being used in the specified topology.");
	case 0xc01e0355u:
		return __tsc(u8"The specified primary surface has a different private-format attribute than the current primary surface.");
	case 0xc01e0356u:
		return __tsc(u8"The specified mode-pruning algorithm is invalid.");
	case 0xc01e0357u:
		return __tsc(u8"The specified monitor-capability origin is invalid.");
	case 0xc01e0358u:
		return __tsc(u8"The specified monitor-frequency range constraint is invalid.");
	case 0xc01e0359u:
		return __tsc(u8"The maximum supported number of present paths has been reached.");
	case 0xc01e035au:
		return __tsc(u8"The miniport requested that augmentation be canceled for the specified source of the specified VidPN's topology.");
	case 0xc01e035bu:
		return __tsc(u8"The specified client type was not recognized.");
	case 0xc01e035cu:
		return __tsc(u8"The client VidPN is not set on this adapter (for example, no user mode-initiated mode changes have taken place on this adapter).");
	case 0xc01e0400u:
		return __tsc(u8"The specified display adapter child device already has an external device connected to it.");
	case 0xc01e0401u:
		return __tsc(u8"The display adapter child device does not support reporting a descriptor.");
	case 0xc01e0430u:
		return __tsc(u8"The display adapter is not linked to any other adapters.");
	case 0xc01e0431u:
		return __tsc(u8"The lead adapter in a linked configuration was not enumerated yet.");
	case 0xc01e0432u:
		return __tsc(u8"Some chain adapters in a linked configuration have not yet been enumerated.");
	case 0xc01e0433u:
		return __tsc(u8"The chain of linked adapters is not ready to start because of an unknown failure.");
	case 0xc01e0434u:
		return __tsc(u8"An attempt was made to start a lead link display adapter when the chain links had not yet started.");
	case 0xc01e0435u:
		return __tsc(u8"An attempt was made to turn on a lead link display adapter when the chain links were turned off.");
	case 0xc01e0436u:
		return __tsc(u8"The adapter link was found in an inconsistent state. Not all adapters are in an expected PNP/power state.");
	case 0xc01e0438u:
		return __tsc(u8"The driver trying to start is not the same as the driver for the posted display adapter.");
	case 0xc01e043bu:
		return __tsc(u8"An operation is being attempted that requires the display adapter to be in a quiescent state.");
	case 0xc01e0500u:
		return __tsc(u8"The driver does not support OPM.");
	case 0xc01e0501u:
		return __tsc(u8"The driver does not support COPP.");
	case 0xc01e0502u:
		return __tsc(u8"The driver does not support UAB.");
	case 0xc01e0503u:
		return __tsc(u8"The specified encrypted parameters are invalid.");
	case 0xc01e0505u:
		return __tsc(u8"The GDI display device passed to this function does not have any active protected outputs.");
	case 0xc01e050bu:
		return __tsc(u8"An internal error caused an operation to fail.");
	case 0xc01e050cu:
		return __tsc(u8"The function failed because the caller passed in an invalid OPM user-mode handle.");
	case 0xc01e050eu:
		return __tsc(u8"A certificate could not be returned because the certificate buffer passed to the function was too small.");
	case 0xc01e050fu:
		return __tsc(u8"DxgkDdiOpmCreateProtectedOutput() could not create a protected output because the video present yarget is in spanning mode.");
	case 0xc01e0510u:
		return __tsc(u8"DxgkDdiOpmCreateProtectedOutput() could not create a protected output because the video present target is in theater mode.");
	case 0xc01e0511u:
		return __tsc(u8"The function call failed because the display adapter's hardware functionality scan (HFS) failed to validate the graphics hardware.");
	case 0xc01e0512u:
		return __tsc(u8"The HDCP SRM passed to this function did not comply with section 5 of the HDCP 1.1 specification.");
	case 0xc01e0513u:
		return __tsc(u8"The protected output cannot enable the HDCP system because it does not support it.");
	case 0xc01e0514u:
		return __tsc(u8"The protected output cannot enable analog copy protection because it does not support it.");
	case 0xc01e0515u:
		return __tsc(u8"The protected output cannot enable the CGMS-A protection technology because it does not support it.");
	case 0xc01e0516u:
		return __tsc(u8"DxgkDdiOPMGetInformation() cannot return the version of the SRM being used because the application never successfully passed an SRM to the protected output.");
	case 0xc01e0517u:
		return __tsc(u8"DxgkDdiOPMConfigureProtectedOutput() cannot enable the specified output protection technology because the output's screen resolution is too high.");
	case 0xc01e0518u:
		return __tsc(u8"DxgkDdiOPMConfigureProtectedOutput() cannot enable HDCP because other physical outputs are using the display adapter's HDCP hardware.");
	case 0xc01e051au:
		return __tsc(u8"The operating system asynchronously destroyed this OPM-protected output because the operating system state changed. This error typically occurs because the monitor PDO associated with this protected output was removed or stopped, the protected output's session became a nonconsole session, or the protected output's desktop became inactive.");
	case 0xc01e051cu:
		return __tsc(u8"The DxgkDdiOPMGetCOPPCompatibleInformation, DxgkDdiOPMGetInformation, or DxgkDdiOPMConfigureProtectedOutput function failed. This error is returned only if a protected output has OPM semantics.\nDxgkDdiOPMGetCOPPCompatibleInformation always returns this error if a protected output has OPM semantics.\nDxgkDdiOPMGetInformation returns this error code if the caller requested COPP-specific information.\nDxgkDdiOPMConfigureProtectedOutput returns this error when the caller tries to use a COPP-specific command.");
	case 0xc01e051du:
		return __tsc(u8"The DxgkDdiOPMGetInformation and DxgkDdiOPMGetCOPPCompatibleInformation functions return this error code if the passed-in sequence number is not the expected sequence number or the passed-in OMAC value is invalid.");
	case 0xc01e051eu:
		return __tsc(u8"The function failed because an unexpected error occurred inside a display driver.");
	case 0xc01e051fu:
		return __tsc(u8"The DxgkDdiOPMGetCOPPCompatibleInformation, DxgkDdiOPMGetInformation, or DxgkDdiOPMConfigureProtectedOutput function failed. This error is returned only if a protected output has COPP semantics.\nDxgkDdiOPMGetCOPPCompatibleInformation returns this error code if the caller requested OPM-specific information.\nDxgkDdiOPMGetInformation always returns this error if a protected output has COPP semantics.\nDxgkDdiOPMConfigureProtectedOutput returns this error when the caller tries to use an OPM-specific command.");
	case 0xc01e0520u:
		return __tsc(u8"The DxgkDdiOPMGetCOPPCompatibleInformation and DxgkDdiOPMConfigureProtectedOutput functions return this error if the display driver does not support the DXGKMDT\\_OPM\\_GET\\_ACP\\_AND\\_CGMSA\\_SIGNALING and DXGKMDT\\_OPM\\_SET\\_ACP\\_AND\\_CGMSA\\_SIGNALING GUIDs.");
	case 0xc01e0521u:
		return __tsc(u8"The DxgkDdiOPMConfigureProtectedOutput function returns this error code if the passed-in sequence number is not the expected sequence number or the passed-in OMAC value is invalid.");
	case 0xc01e0580u:
		return __tsc(u8"The monitor connected to the specified video output does not have an I2C bus.");
	case 0xc01e0581u:
		return __tsc(u8"No device on the I2C bus has the specified address.");
	case 0xc01e0582u:
		return __tsc(u8"An error occurred while transmitting data to the device on the I2C bus.");
	case 0xc01e0583u:
		return __tsc(u8"An error occurred while receiving data from the device on the I2C bus.");
	case 0xc01e0584u:
		return __tsc(u8"The monitor does not support the specified VCP code.");
	case 0xc01e0585u:
		return __tsc(u8"The data received from the monitor is invalid.");
	case 0xc01e0586u:
		return __tsc(u8"A function call failed because a monitor returned an invalid timing status byte when the operating system used the DDC/CI get timing report and timing message command to get a timing report from a monitor.");
	case 0xc01e0587u:
		return __tsc(u8"A monitor returned a DDC/CI capabilities string that did not comply with the ACCESS.bus 3.0, DDC/CI 1.1, or MCCS 2 Revision 1 specification.");
	case 0xc01e0588u:
		return __tsc(u8"An internal error caused an operation to fail.");
	case 0xc01e0589u:
		return __tsc(u8"An operation failed because a DDC/CI message had an invalid value in its command field.");
	case 0xc01e058au:
		return __tsc(u8"This error occurred because a DDC/CI message had an invalid value in its length field.");
	case 0xc01e058bu:
		return __tsc(u8"This error occurred because the value in a DDC/CI message's checksum field did not match the message's computed checksum value. This error implies that the data was corrupted while it was being transmitted from a monitor to a computer.");
	case 0xc01e058cu:
		return __tsc(u8"This function failed because an invalid monitor handle was passed to it.");
	case 0xc01e058du:
		return __tsc(u8"The operating system asynchronously destroyed the monitor that corresponds to this handle because the operating system's state changed. This error typically occurs because the monitor PDO associated with this handle was removed or stopped, or a display mode change occurred. A display mode change occurs when Windows sends a WM\\_DISPLAYCHANGE message to applications.");
	case 0xc01e05e0u:
		return __tsc(u8"This function can be used only if a program is running in the local console session. It cannot be used if a program is running on a remote desktop session or on a terminal server session.");
	case 0xc01e05e1u:
		return __tsc(u8"This function cannot find an actual GDI display device that corresponds to the specified GDI display device name.");
	case 0xc01e05e2u:
		return __tsc(u8"The function failed because the specified GDI display device was not attached to the Windows desktop.");
	case 0xc01e05e3u:
		return __tsc(u8"This function does not support GDI mirroring display devices because GDI mirroring display devices do not have any physical monitors associated with them.");
	case 0xc01e05e4u:
		return __tsc(u8"The function failed because an invalid pointer parameter was passed to it. A pointer parameter is invalid if it is null, is not correctly aligned, or points to an invalid address or to a kernel mode address.");
	case 0xc01e05e5u:
		return __tsc(u8"This function failed because the GDI device passed to it did not have a monitor associated with it.");
	case 0xc01e05e6u:
		return __tsc(u8"An array passed to the function cannot hold all of the data that the function must copy into the array.");
	case 0xc01e05e7u:
		return __tsc(u8"An internal error caused an operation to fail.");
	case 0xc01e05e8u:
		return __tsc(u8"The function failed because the current session is changing its type. This function cannot be called when the current session is changing its type. Three types of sessions currently exist: console, disconnected, and remote (RDP or ICA).");
	case 0xc0210000u:
		return __tsc(u8"The volume must be unlocked before it can be used.");
	case 0xc0210001u:
		return __tsc(u8"The volume is fully decrypted and no key is available.");
	case 0xc0210002u:
		return __tsc(u8"The control block for the encrypted volume is not valid.");
	case 0xc0210003u:
		return __tsc(u8"Not enough free space remains on the volume to allow encryption.");
	case 0xc0210004u:
		return __tsc(u8"The partition cannot be encrypted because the file system is not supported.");
	case 0xc0210005u:
		return __tsc(u8"The file system is inconsistent. Run the Check Disk utility.");
	case 0xc0210006u:
		return __tsc(u8"The file system does not extend to the end of the volume.");
	case 0xc0210007u:
		return __tsc(u8"This operation cannot be performed while a file system is mounted on the volume.");
	case 0xc0210008u:
		return __tsc(u8"BitLocker Drive Encryption is not included with this version of Windows.");
	case 0xc0210009u:
		return __tsc(u8"The requested action was denied by the FVE control engine.");
	case 0xc021000au:
		return __tsc(u8"The data supplied is malformed.");
	case 0xc021000bu:
		return __tsc(u8"The volume is not bound to the system.");
	case 0xc021000cu:
		return __tsc(u8"The volume specified is not a data volume.");
	case 0xc021000du:
		return __tsc(u8"A read operation failed while converting the volume.");
	case 0xc021000eu:
		return __tsc(u8"A write operation failed while converting the volume.");
	case 0xc021000fu:
		return __tsc(u8"The control block for the encrypted volume was updated by another thread. Try again.");
	case 0xc0210010u:
		return __tsc(u8"The volume encryption algorithm cannot be used on this sector size.");
	case 0xc0210011u:
		return __tsc(u8"BitLocker recovery authentication failed.");
	case 0xc0210012u:
		return __tsc(u8"The volume specified is not the boot operating system volume.");
	case 0xc0210013u:
		return __tsc(u8"The BitLocker startup key or recovery password could not be read from external media.");
	case 0xc0210014u:
		return __tsc(u8"The BitLocker startup key or recovery password file is corrupt or invalid.");
	case 0xc0210015u:
		return __tsc(u8"The BitLocker encryption key could not be obtained from the startup key or the recovery password.");
	case 0xc0210016u:
		return __tsc(u8"The TPM is disabled.");
	case 0xc0210017u:
		return __tsc(u8"The authorization data for the SRK of the TPM is not zero.");
	case 0xc0210018u:
		return __tsc(u8"The system boot information changed or the TPM locked out access to BitLocker encryption keys until the computer is restarted.");
	case 0xc0210019u:
		return __tsc(u8"The BitLocker encryption key could not be obtained from the TPM.");
	case 0xc021001au:
		return __tsc(u8"The BitLocker encryption key could not be obtained from the TPM and PIN.");
	case 0xc021001bu:
		return __tsc(u8"A boot application hash does not match the hash computed when BitLocker was turned on.");
	case 0xc021001cu:
		return __tsc(u8"The Boot Configuration Data (BCD) settings are not supported or have changed because BitLocker was enabled.");
	case 0xc021001du:
		return __tsc(u8"Boot debugging is enabled. Run Windows Boot Configuration Data Store Editor (bcdedit.exe) to turn it off.");
	case 0xc021001eu:
		return __tsc(u8"The BitLocker encryption key could not be obtained.");
	case 0xc021001fu:
		return __tsc(u8"The metadata disk region pointer is incorrect.");
	case 0xc0210020u:
		return __tsc(u8"The backup copy of the metadata is out of date.");
	case 0xc0210021u:
		return __tsc(u8"No action was taken because a system restart is required.");
	case 0xc0210022u:
		return __tsc(u8"No action was taken because BitLocker Drive Encryption is in RAW access mode.");
	case 0xc0210023u:
		return __tsc(u8"BitLocker Drive Encryption cannot enter RAW access mode for this volume.");
	case 0xc0210024u:
	case 0xc0210025u:
		return __tsc(u8"");
	case 0xc0210026u:
		return __tsc(u8"This feature of BitLocker Drive Encryption is not included with this version of Windows.");
	case 0xc0210027u:
		return __tsc(u8"Group policy does not permit turning off BitLocker Drive Encryption on roaming data volumes.");
	case 0xc0210028u:
		return __tsc(u8"Bitlocker Drive Encryption failed to recover from aborted conversion. This could be due to either all conversion logs being corrupted or the media being write-protected.");
	case 0xc0210029u:
		return __tsc(u8"The requested virtualization size is too big.");
	case 0xc021002au:
		return __tsc(u8"");
	case 0xc0210030u:
		return __tsc(u8"The drive is too small to be protected using BitLocker Drive Encryption.");
	case 0xc0210031u:
	case 0xc0210032u:
	case 0xc0210033u:
	case 0xc0210034u:
	case 0xc0210035u:
	case 0xc0210036u:
	case 0xc0210037u:
	case 0xc0210038u:
	case 0xc0210039u:
	case 0xc021003au:
	case 0xc021003bu:
	case 0xc021003cu:
	case 0xc021003du:
	case 0xc021003eu:
	case 0xc021003fu:
	case 0xc0210040u:
	case 0xc0210041u:
	case 0xc0210042u:
	case 0xc0210043u:
	case 0xc0210044u:
		return __tsc(u8"");
	case 0xc0220001u:
		return __tsc(u8"The callout does not exist.");
	case 0xc0220002u:
		return __tsc(u8"The filter condition does not exist.");
	case 0xc0220003u:
		return __tsc(u8"The filter does not exist.");
	case 0xc0220004u:
		return __tsc(u8"The layer does not exist.");
	case 0xc0220005u:
		return __tsc(u8"The provider does not exist.");
	case 0xc0220006u:
		return __tsc(u8"The provider context does not exist.");
	case 0xc0220007u:
		return __tsc(u8"The sublayer does not exist.");
	case 0xc0220008u:
		return __tsc(u8"The object does not exist.");
	case 0xc0220009u:
		return __tsc(u8"An object with that GUID or LUID already exists.");
	case 0xc022000au:
		return __tsc(u8"The object is referenced by other objects and cannot be deleted.");
	case 0xc022000bu:
		return __tsc(u8"The call is not allowed from within a dynamic session.");
	case 0xc022000cu:
		return __tsc(u8"The call was made from the wrong session and cannot be completed.");
	case 0xc022000du:
		return __tsc(u8"The call must be made from within an explicit transaction.");
	case 0xc022000eu:
		return __tsc(u8"The call is not allowed from within an explicit transaction.");
	case 0xc022000fu:
		return __tsc(u8"The explicit transaction has been forcibly canceled.");
	case 0xc0220010u:
		return __tsc(u8"The session has been canceled.");
	case 0xc0220011u:
		return __tsc(u8"The call is not allowed from within a read-only transaction.");
	case 0xc0220012u:
		return __tsc(u8"The call timed out while waiting to acquire the transaction lock.");
	case 0xc0220013u:
		return __tsc(u8"The collection of network diagnostic events is disabled.");
	case 0xc0220014u:
		return __tsc(u8"The operation is not supported by the specified layer.");
	case 0xc0220015u:
		return __tsc(u8"The call is allowed for kernel-mode callers only.");
	case 0xc0220016u:
		return __tsc(u8"The call tried to associate two objects with incompatible lifetimes.");
	case 0xc0220017u:
		return __tsc(u8"The object is built-in and cannot be deleted.");
	case 0xc0220018u:
		return __tsc(u8"The maximum number of boot-time filters has been reached.");
	case 0xc0220019u:
		return __tsc(u8"A notification could not be delivered because a message queue has reached maximum capacity.");
	case 0xc022001au:
		return __tsc(u8"The traffic parameters do not match those for the security association context.");
	case 0xc022001bu:
		return __tsc(u8"The call is not allowed for the current security association state.");
	case 0xc022001cu:
		return __tsc(u8"A required pointer is null.");
	case 0xc022001du:
		return __tsc(u8"An enumerator is not valid.");
	case 0xc022001eu:
		return __tsc(u8"The flags field contains an invalid value.");
	case 0xc022001fu:
		return __tsc(u8"A network mask is not valid.");
	case 0xc0220020u:
		return __tsc(u8"An FWP\\_RANGE is not valid.");
	case 0xc0220021u:
		return __tsc(u8"The time interval is not valid.");
	case 0xc0220022u:
		return __tsc(u8"An array that must contain at least one element has a zero length.");
	case 0xc0220023u:
		return __tsc(u8"The displayData.name field cannot be null.");
	case 0xc0220024u:
		return __tsc(u8"The action type is not one of the allowed action types for a filter.");
	case 0xc0220025u:
		return __tsc(u8"The filter weight is not valid.");
	case 0xc0220026u:
		return __tsc(u8"A filter condition contains a match type that is not compatible with the operands.");
	case 0xc0220027u:
		return __tsc(u8"An FWP\\_VALUE or FWPM\\_CONDITION\\_VALUE is of the wrong type.");
	case 0xc0220028u:
		return __tsc(u8"An integer value is outside the allowed range.");
	case 0xc0220029u:
		return __tsc(u8"A reserved field is nonzero.");
	case 0xc022002au:
		return __tsc(u8"A filter cannot contain multiple conditions operating on a single field.");
	case 0xc022002bu:
		return __tsc(u8"A policy cannot contain the same keying module more than once.");
	case 0xc022002cu:
		return __tsc(u8"The action type is not compatible with the layer.");
	case 0xc022002du:
		return __tsc(u8"The action type is not compatible with the sublayer.");
	case 0xc022002eu:
		return __tsc(u8"The raw context or the provider context is not compatible with the layer.");
	case 0xc022002fu:
		return __tsc(u8"The raw context or the provider context is not compatible with the callout.");
	case 0xc0220030u:
		return __tsc(u8"The authentication method is not compatible with the policy type.");
	case 0xc0220031u:
		return __tsc(u8"The Diffie-Hellman group is not compatible with the policy type.");
	case 0xc0220032u:
		return __tsc(u8"An IKE policy cannot contain an Extended Mode policy.");
	case 0xc0220033u:
		return __tsc(u8"The enumeration template or subscription will never match any objects.");
	case 0xc0220034u:
		return __tsc(u8"The provider context is of the wrong type.");
	case 0xc0220035u:
		return __tsc(u8"The parameter is incorrect.");
	case 0xc0220036u:
		return __tsc(u8"The maximum number of sublayers has been reached.");
	case 0xc0220037u:
		return __tsc(u8"The notification function for a callout returned an error.");
	case 0xc0220038u:
		return __tsc(u8"The IPsec authentication configuration is not compatible with the authentication type.");
	case 0xc0220039u:
		return __tsc(u8"The IPsec cipher configuration is not compatible with the cipher type.");
	case 0xc022003au:
	case 0xc022003bu:
		return __tsc(u8"");
	case 0xc022003cu:
		return __tsc(u8"A policy cannot contain the same auth method more than once.");
	case 0xc022003du:
	case 0xc022003eu:
	case 0xc022003fu:
	case 0xc0220040u:
	case 0xc0220041u:
	case 0xc0220042u:
	case 0xc0220043u:
	case 0xc0220044u:
		return __tsc(u8"");
	case 0xc0220100u:
		return __tsc(u8"The TCP/IP stack is not ready.");
	case 0xc0220101u:
		return __tsc(u8"The injection handle is being closed by another thread.");
	case 0xc0220102u:
		return __tsc(u8"The injection handle is stale.");
	case 0xc0220103u:
		return __tsc(u8"The classify cannot be pended.");
	case 0xc0220104u:
		return __tsc(u8"");
	case 0xc0230002u:
		return __tsc(u8"The binding to the network interface is being closed.");
	case 0xc0230004u:
		return __tsc(u8"An invalid version was specified.");
	case 0xc0230005u:
		return __tsc(u8"An invalid characteristics table was used.");
	case 0xc0230006u:
		return __tsc(u8"Failed to find the network interface or the network interface is not ready.");
	case 0xc0230007u:
		return __tsc(u8"Failed to open the network interface.");
	case 0xc0230008u:
		return __tsc(u8"The network interface has encountered an internal unrecoverable failure.");
	case 0xc0230009u:
		return __tsc(u8"The multicast list on the network interface is full.");
	case 0xc023000au:
		return __tsc(u8"An attempt was made to add a duplicate multicast address to the list.");
	case 0xc023000bu:
		return __tsc(u8"At attempt was made to remove a multicast address that was never added.");
	case 0xc023000cu:
		return __tsc(u8"The network interface aborted the request.");
	case 0xc023000du:
		return __tsc(u8"The network interface cannot process the request because it is being reset.");
	case 0xc023000fu:
		return __tsc(u8"An attempt was made to send an invalid packet on a network interface.");
	case 0xc0230010u:
		return __tsc(u8"The specified request is not a valid operation for the target device.");
	case 0xc0230011u:
		return __tsc(u8"The network interface is not ready to complete this operation.");
	case 0xc0230014u:
		return __tsc(u8"The length of the buffer submitted for this operation is not valid.");
	case 0xc0230015u:
		return __tsc(u8"The data used for this operation is not valid.");
	case 0xc0230016u:
		return __tsc(u8"The length of the submitted buffer for this operation is too small.");
	case 0xc0230017u:
		return __tsc(u8"The network interface does not support this object identifier.");
	case 0xc0230018u:
		return __tsc(u8"The network interface has been removed.");
	case 0xc0230019u:
		return __tsc(u8"The network interface does not support this media type.");
	case 0xc023001au:
		return __tsc(u8"An attempt was made to remove a token ring group address that is in use by other components.");
	case 0xc023001bu:
		return __tsc(u8"An attempt was made to map a file that cannot be found.");
	case 0xc023001cu:
		return __tsc(u8"An error occurred while NDIS tried to map the file.");
	case 0xc023001du:
		return __tsc(u8"An attempt was made to map a file that is already mapped.");
	case 0xc023001eu:
		return __tsc(u8"An attempt to allocate a hardware resource failed because the resource is used by another component.");
	case 0xc023001fu:
		return __tsc(u8"The I/O operation failed because the network media is disconnected or the wireless access point is out of range.");
	case 0xc0230022u:
		return __tsc(u8"The network address used in the request is invalid.");
	case 0xc023002au:
		return __tsc(u8"The offload operation on the network interface has been paused.");
	case 0xc023002bu:
		return __tsc(u8"The network interface was not found.");
	case 0xc023002cu:
		return __tsc(u8"The revision number specified in the structure is not supported.");
	case 0xc023002du:
		return __tsc(u8"The specified port does not exist on this network interface.");
	case 0xc023002eu:
		return __tsc(u8"The current state of the specified port on this network interface does not support the requested operation.");
	case 0xc023002fu:
		return __tsc(u8"The miniport adapter is in a lower power state.");
	case 0xc0230030u:
	case 0xc0230031u:
		return __tsc(u8"");
	case 0xc02300bbu:
		return __tsc(u8"The network interface does not support this request.");
	case 0xc023100fu:
		return __tsc(u8"The TCP connection is not offloadable because of a local policy setting.");
	case 0xc0231012u:
		return __tsc(u8"The TCP connection is not offloadable by the Chimney offload target.");
	case 0xc0231013u:
		return __tsc(u8"The IP Path object is not in an offloadable state.");
	case 0xc0232000u:
		return __tsc(u8"The wireless LAN interface is in auto-configuration mode and does not support the requested parameter change operation.");
	case 0xc0232001u:
		return __tsc(u8"The wireless LAN interface is busy and cannot perform the requested operation.");
	case 0xc0232002u:
		return __tsc(u8"The wireless LAN interface is power down and does not support the requested operation.");
	case 0xc0232003u:
		return __tsc(u8"The list of wake on LAN patterns is full.");
	case 0xc0232004u:
		return __tsc(u8"The list of low power protocol offloads is full.");
	case 0xc0232005u:
	case 0xc0232006u:
	case 0xc0232007u:
	case 0xc0232008u:
	case 0xc0240000u:
	case 0xc0240001u:
	case 0xc0240002u:
	case 0xc0240003u:
	case 0xc0240004u:
	case 0xc0240005u:
	case 0xc0240006u:
	case 0xc0240007u:
	case 0xc0290000u:
	case 0xc0290001u:
	case 0xc0290002u:
	case 0xc0290003u:
	case 0xc0290004u:
	case 0xc0290005u:
	case 0xc0290006u:
	case 0xc0290007u:
	case 0xc0290008u:
	case 0xc0290009u:
	case 0xc029000au:
	case 0xc029000bu:
	case 0xc029000cu:
	case 0xc029000du:
	case 0xc029000eu:
	case 0xc029000fu:
	case 0xc0290010u:
	case 0xc0290011u:
	case 0xc0290012u:
	case 0xc0290013u:
	case 0xc0290014u:
	case 0xc0290015u:
	case 0xc0290016u:
	case 0xc0290017u:
	case 0xc0290018u:
	case 0xc0290019u:
	case 0xc029001au:
	case 0xc029001bu:
	case 0xc029001cu:
	case 0xc029001du:
	case 0xc029001eu:
	case 0xc029001fu:
	case 0xc0290020u:
	case 0xc0290021u:
	case 0xc0290022u:
	case 0xc0290023u:
	case 0xc0290024u:
	case 0xc0290025u:
	case 0xc0290026u:
	case 0xc0290027u:
	case 0xc0290028u:
	case 0xc0290029u:
	case 0xc029002au:
	case 0xc029002bu:
	case 0xc029002cu:
	case 0xc029002du:
	case 0xc029002eu:
	case 0xc029002fu:
	case 0xc0290030u:
	case 0xc0290031u:
	case 0xc0290032u:
	case 0xc0290033u:
	case 0xc0290034u:
	case 0xc0290035u:
	case 0xc0290036u:
	case 0xc0290037u:
	case 0xc0290038u:
	case 0xc0290039u:
	case 0xc029003au:
	case 0xc029003bu:
	case 0xc029003cu:
	case 0xc029003du:
	case 0xc029003eu:
	case 0xc029003fu:
	case 0xc0290040u:
	case 0xc0290041u:
	case 0xc0290042u:
	case 0xc0290043u:
	case 0xc0290044u:
	case 0xc0290045u:
	case 0xc0290046u:
	case 0xc0290047u:
	case 0xc0290048u:
	case 0xc0290049u:
	case 0xc029004au:
	case 0xc029004bu:
	case 0xc029004cu:
	case 0xc029004du:
	case 0xc029004eu:
	case 0xc029004fu:
	case 0xc0290050u:
	case 0xc0290051u:
	case 0xc0290052u:
	case 0xc0290053u:
	case 0xc0290054u:
	case 0xc0290055u:
	case 0xc0290056u:
	case 0xc0290057u:
	case 0xc0290058u:
	case 0xc0290059u:
	case 0xc029005au:
	case 0xc029005bu:
	case 0xc029005cu:
	case 0xc029005du:
	case 0xc029005eu:
	case 0xc029005fu:
	case 0xc0290061u:
	case 0xc0290062u:
	case 0xc0290063u:
	case 0xc0290081u:
	case 0xc0290082u:
	case 0xc0290083u:
	case 0xc0290084u:
	case 0xc0290085u:
	case 0xc0290087u:
	case 0xc0290088u:
	case 0xc0290089u:
	case 0xc029008au:
	case 0xc029008bu:
	case 0xc029008cu:
	case 0xc029008du:
	case 0xc029008eu:
	case 0xc029008fu:
	case 0xc0290090u:
	case 0xc0290092u:
	case 0xc0290095u:
	case 0xc0290096u:
	case 0xc0290097u:
	case 0xc0290098u:
	case 0xc029009au:
	case 0xc029009bu:
	case 0xc029009cu:
	case 0xc029009du:
	case 0xc029009fu:
	case 0xc02900a0u:
	case 0xc02900a1u:
	case 0xc02900a2u:
	case 0xc02900a3u:
	case 0xc02900a4u:
	case 0xc02900a5u:
	case 0xc02900a6u:
	case 0xc02900a7u:
	case 0xc0290100u:
	case 0xc0290101u:
	case 0xc0290103u:
	case 0xc029010bu:
	case 0xc0290119u:
	case 0xc0290120u:
	case 0xc0290121u:
	case 0xc0290123u:
	case 0xc0290124u:
	case 0xc0290125u:
	case 0xc0290126u:
	case 0xc0290127u:
	case 0xc0290128u:
	case 0xc029012du:
	case 0xc029012eu:
	case 0xc029012fu:
	case 0xc0290130u:
	case 0xc0290131u:
	case 0xc0290142u:
	case 0xc0290143u:
	case 0xc0290144u:
	case 0xc0290145u:
	case 0xc0290146u:
	case 0xc0290147u:
	case 0xc0290148u:
	case 0xc0290149u:
	case 0xc029014au:
	case 0xc029014bu:
	case 0xc029014cu:
	case 0xc0290150u:
	case 0xc0290151u:
	case 0xc0290152u:
	case 0xc0290153u:
	case 0xc0290154u:
	case 0xc0290155u:
	case 0xc0290400u:
	case 0xc0290401u:
	case 0xc0290402u:
	case 0xc0290403u:
	case 0xc0290404u:
	case 0xc0290800u:
	case 0xc0290801u:
	case 0xc0290802u:
	case 0xc0290803u:
	case 0xc0291001u:
	case 0xc0291002u:
	case 0xc0291003u:
	case 0xc0291004u:
	case 0xc0291005u:
	case 0xc0291006u:
	case 0xc0292000u:
	case 0xc0292001u:
	case 0xc0292002u:
	case 0xc0292003u:
	case 0xc0292004u:
	case 0xc0292005u:
	case 0xc0292006u:
	case 0xc0292007u:
	case 0xc0292008u:
	case 0xc0292009u:
	case 0xc029200au:
	case 0xc029200bu:
	case 0xc029200cu:
	case 0xc029200du:
	case 0xc029200eu:
	case 0xc029200fu:
	case 0xc0292010u:
	case 0xc0292011u:
	case 0xc0292012u:
	case 0xc0292013u:
	case 0xc0292014u:
	case 0xc0292015u:
	case 0xc0292016u:
	case 0xc0292017u:
	case 0xc0292018u:
	case 0xc0292019u:
	case 0xc029201au:
	case 0xc029201bu:
	case 0xc029201cu:
	case 0xc029201du:
	case 0xc029201eu:
	case 0xc029201fu:
	case 0xc0292020u:
	case 0xc0292021u:
	case 0xc0292022u:
	case 0xc0293002u:
	case 0xc0293003u:
	case 0xc0293004u:
	case 0xc0293005u:
	case 0xc0294000u:
	case 0xc0350002u:
	case 0xc0350003u:
	case 0xc0350004u:
	case 0xc0350005u:
	case 0xc0350006u:
	case 0xc0350007u:
	case 0xc0350008u:
	case 0xc0350009u:
	case 0xc035000au:
	case 0xc035000bu:
	case 0xc035000cu:
	case 0xc035000du:
	case 0xc035000eu:
	case 0xc0350011u:
	case 0xc0350012u:
	case 0xc0350013u:
	case 0xc0350014u:
	case 0xc0350015u:
	case 0xc0350016u:
	case 0xc0350017u:
	case 0xc0350018u:
	case 0xc0350019u:
	case 0xc035001au:
	case 0xc035001bu:
	case 0xc035001cu:
	case 0xc035001du:
	case 0xc035001eu:
	case 0xc0350033u:
	case 0xc0350038u:
	case 0xc035003cu:
	case 0xc035003du:
	case 0xc035003eu:
	case 0xc035003fu:
	case 0xc0350041u:
	case 0xc0350050u:
	case 0xc0350051u:
	case 0xc0350055u:
	case 0xc0350057u:
	case 0xc0350058u:
	case 0xc0350060u:
	case 0xc035006fu:
	case 0xc0350070u:
	case 0xc0350071u:
	case 0xc0350072u:
	case 0xc0350073u:
	case 0xc0350074u:
	case 0xc0350075u:
	case 0xc0350076u:
	case 0xc0350077u:
	case 0xc0350079u:
	case 0xc0350080u:
	case 0xc0350081u:
	case 0xc0350082u:
	case 0xc0350083u:
	case 0xc0350084u:
	case 0xc0350085u:
	case 0xc0351000u:
		return __tsc(u8"");
	case 0xc0360001u:
		return __tsc(u8"The SPI in the packet does not match a valid IPsec SA.");
	case 0xc0360002u:
		return __tsc(u8"The packet was received on an IPsec SA whose lifetime has expired.");
	case 0xc0360003u:
		return __tsc(u8"The packet was received on an IPsec SA that does not match the packet characteristics.");
	case 0xc0360004u:
		return __tsc(u8"The packet sequence number replay check failed.");
	case 0xc0360005u:
		return __tsc(u8"The IPsec header and/or trailer in the packet is invalid.");
	case 0xc0360006u:
		return __tsc(u8"The IPsec integrity check failed.");
	case 0xc0360007u:
		return __tsc(u8"IPsec dropped a clear text packet.");
	case 0xc0360008u:
		return __tsc(u8"IPsec dropped an incoming ESP packet in authenticated firewall mode. This drop is benign.");
	case 0xc0360009u:
		return __tsc(u8"IPsec dropped a packet due to DOS throttle.");
	case 0xc0368000u:
		return __tsc(u8"IPsec Dos Protection matched an explicit block rule.");
	case 0xc0368001u:
		return __tsc(u8"IPsec Dos Protection received an IPsec specific multicast packet which is not allowed.");
	case 0xc0368002u:
		return __tsc(u8"IPsec Dos Protection received an incorrectly formatted packet.");
	case 0xc0368003u:
		return __tsc(u8"IPsec Dos Protection failed to lookup state.");
	case 0xc0368004u:
		return __tsc(u8"IPsec Dos Protection failed to create state because there are already maximum number of entries allowed by policy.");
	case 0xc0368005u:
		return __tsc(u8"IPsec Dos Protection received an IPsec negotiation packet for a keying module which is not allowed by policy.");
	case 0xc0368006u:
		return __tsc(u8"IPsec Dos Protection failed to create per internal IP ratelimit queue because there is already maximum number of queues allowed by policy.");
	case 0xc0370001u:
	case 0xc0370002u:
	case 0xc0370003u:
	case 0xc0370004u:
	case 0xc0370005u:
	case 0xc0370006u:
	case 0xc0370007u:
	case 0xc0370008u:
	case 0xc0370009u:
	case 0xc037000au:
	case 0xc037000bu:
	case 0xc037000cu:
	case 0xc037000du:
	case 0xc037000eu:
	case 0xc037000fu:
	case 0xc0370010u:
	case 0xc0370011u:
	case 0xc0370012u:
	case 0xc0370013u:
	case 0xc0370014u:
	case 0xc0370015u:
	case 0xc0370016u:
	case 0xc0370017u:
	case 0xc0370018u:
	case 0xc0370019u:
	case 0xc037001au:
	case 0xc037001bu:
	case 0xc037001cu:
	case 0xc037001du:
	case 0xc037001eu:
	case 0xc037001fu:
	case 0xc0370020u:
	case 0xc0370021u:
	case 0xc0370022u:
	case 0xc0370023u:
	case 0xc0370024u:
	case 0xc0370025u:
	case 0xc0370026u:
	case 0xc0370027u:
	case 0xc0370028u:
	case 0xc0370029u:
	case 0xc037002au:
	case 0xc037002bu:
	case 0xc037002cu:
	case 0xc037002du:
	case 0xc037002eu:
	case 0xc037002fu:
	case 0xc0370030u:
	case 0xc0370600u:
	case 0xc0380001u:
	case 0xc0380002u:
	case 0xc0380003u:
	case 0xc0380004u:
	case 0xc0380005u:
	case 0xc0380006u:
	case 0xc0380007u:
	case 0xc0380008u:
	case 0xc0380009u:
	case 0xc038000au:
	case 0xc038000bu:
	case 0xc038000cu:
	case 0xc038000du:
	case 0xc038000eu:
	case 0xc038000fu:
	case 0xc0380010u:
	case 0xc0380011u:
	case 0xc0380012u:
	case 0xc0380013u:
	case 0xc0380014u:
	case 0xc0380015u:
	case 0xc0380016u:
	case 0xc0380017u:
	case 0xc0380018u:
	case 0xc0380019u:
	case 0xc038001au:
	case 0xc038001bu:
	case 0xc038001cu:
	case 0xc038001du:
	case 0xc038001eu:
	case 0xc038001fu:
	case 0xc0380020u:
	case 0xc0380021u:
	case 0xc0380022u:
	case 0xc0380023u:
	case 0xc0380024u:
	case 0xc0380025u:
	case 0xc0380026u:
	case 0xc0380027u:
	case 0xc0380028u:
	case 0xc0380029u:
	case 0xc038002au:
	case 0xc038002bu:
	case 0xc038002cu:
	case 0xc038002du:
	case 0xc038002eu:
	case 0xc038002fu:
	case 0xc0380030u:
	case 0xc0380031u:
	case 0xc0380032u:
	case 0xc0380033u:
	case 0xc0380034u:
	case 0xc0380035u:
	case 0xc0380036u:
	case 0xc0380037u:
	case 0xc0380038u:
	case 0xc0380039u:
	case 0xc038003au:
	case 0xc038003bu:
	case 0xc038003cu:
	case 0xc038003du:
	case 0xc038003eu:
	case 0xc038003fu:
	case 0xc0380040u:
	case 0xc0380041u:
	case 0xc0380042u:
	case 0xc0380043u:
	case 0xc0380044u:
	case 0xc0380045u:
	case 0xc0380046u:
	case 0xc0380047u:
	case 0xc0380048u:
	case 0xc0380049u:
	case 0xc038004au:
	case 0xc038004bu:
	case 0xc038004cu:
	case 0xc038004du:
	case 0xc038004eu:
	case 0xc038004fu:
	case 0xc0380050u:
	case 0xc0380051u:
	case 0xc0380052u:
	case 0xc0380053u:
	case 0xc0380054u:
	case 0xc0380055u:
	case 0xc0380056u:
	case 0xc0380057u:
	case 0xc0380058u:
	case 0xc0380059u:
	case 0xc038005au:
		return __tsc(u8"");
	case 0xc038005bu:
		return __tsc(u8"The system does not support mirrored volumes.");
	case 0xc038005cu:
		return __tsc(u8"The system does not support RAID-5 volumes.");
	case 0xc0390002u:
	case 0xc03a0001u:
	case 0xc03a0002u:
	case 0xc03a0003u:
	case 0xc03a0004u:
	case 0xc03a0005u:
	case 0xc03a0006u:
	case 0xc03a0007u:
	case 0xc03a0008u:
	case 0xc03a0009u:
	case 0xc03a000au:
	case 0xc03a000bu:
	case 0xc03a000cu:
	case 0xc03a000du:
	case 0xc03a000eu:
	case 0xc03a000fu:
	case 0xc03a0010u:
	case 0xc03a0011u:
	case 0xc03a0012u:
	case 0xc03a0013u:
		return __tsc(u8"");
	case 0xc03a0014u:
		return __tsc(u8"A virtual disk support provider for the specified file was not found.");
	case 0xc03a0015u:
		return __tsc(u8"The specified disk is not a virtual disk.");
	case 0xc03a0016u:
		return __tsc(u8"The chain of virtual hard disks is inaccessible. The process has not been granted access rights to the parent virtual hard disk for the differencing disk.");
	case 0xc03a0017u:
		return __tsc(u8"The chain of virtual hard disks is corrupted. There is a mismatch in the virtual sizes of the parent virtual hard disk and differencing disk.");
	case 0xc03a0018u:
		return __tsc(u8"The chain of virtual hard disks is corrupted. A differencing disk is indicated in its own parent chain.");
	case 0xc03a0019u:
		return __tsc(u8"The chain of virtual hard disks is inaccessible. There was an error opening a virtual hard disk further up the chain.");
	case 0xc03a001au:
	case 0xc03a001bu:
	case 0xc03a001cu:
	case 0xc03a001du:
	case 0xc03a001eu:
	case 0xc03a001fu:
	case 0xc03a0020u:
	case 0xc03a0021u:
	case 0xc03a0022u:
	case 0xc03a0023u:
	case 0xc03a0024u:
	case 0xc03a0028u:
	case 0xc03a0029u:
	case 0xc03a002au:
	case 0xc03a0030u:
	case 0xc03a0031u:
	case 0xc03a0032u:
	case 0xc03a0033u:
	case 0xc0400001u:
	case 0xc0400002u:
	case 0xc0400003u:
	case 0xc0400004u:
	case 0xc0400005u:
	case 0xc0400006u:
	case 0xc0410001u:
	case 0xc0410002u:
	case 0xc0410003u:
	case 0xc0410004u:
	case 0xc0420001u:
	case 0xc0420002u:
	case 0xc0420003u:
	case 0xc0420004u:
	case 0xc0420005u:
	case 0xc0420006u:
	case 0xc0420007u:
	case 0xc0420008u:
	case 0xc0420009u:
	case 0xc042000au:
	case 0xc042000bu:
	case 0xc042000cu:
	case 0xc042000du:
	case 0xc042000eu:
	case 0xc042000fu:
	case 0xc0420010u:
	case 0xc0420011u:
	case 0xc0421000u:
	case 0xc0430001u:
	case 0xc0430002u:
	case 0xc0430003u:
	case 0xc0430004u:
	case 0xc0430005u:
	case 0xc0430007u:
	case 0xc0430008u:
	case 0xc0430009u:
	case 0xc043000au:
	case 0xc043000bu:
	case 0xc043000cu:
	case 0xc043000du:
	case 0xc043000eu:
	case 0xc043000fu:
	case 0xc0430010u:
	case 0xc0440001u:
	case 0xc0440002u:
	case 0xc0440003u:
	case 0xc0440004u:
	case 0xc0440005u:
	case 0xc0450000u:
	case 0xc0450001u:
	case 0xc0460001u:
	case 0xc0460002u:
	case 0xc0460003u:
	case 0xc0460004u:
	case 0xc0460005u:
	case 0xc0460006u:
	case 0xc0460007u:
	case 0xc0460008u:
	case 0xc0500003u:
	case 0xc0500004u:
	case 0xc0500005u:
	case 0xc0510001u:
	case 0xc05c0000u:
	case 0xc05cff00u:
	case 0xc05cff01u:
	case 0xc05cff02u:
	case 0xc05cff03u:
	case 0xc05cff04u:
	case 0xc05cff05u:
	case 0xc05cff06u:
	case 0xc05cff07u:
	case 0xc05cff08u:
	case 0xc05cff09u:
	case 0xc05cff0au:
	case 0xc05cff0bu:
	case 0xc05cff0cu:
		return __tsc(u8"");
	case 0xc05d0000u:
		return __tsc(u8"Returned in response to a client negotiate request when the server does not support any of the hash algorithms in the request.");
	case 0xc05d0001u:
		return __tsc(u8"The current cluster functional level does not support this SMB dialect.");
	case 0xc05d0002u:
	case 0xc05d0003u:
	case 0xc05d0004u:
	case 0xc0e70001u:
	case 0xc0e70003u:
	case 0xc0e70004u:
	case 0xc0e70006u:
	case 0xc0e70007u:
	case 0xc0e70009u:
	case 0xc0e7000au:
	case 0xc0e7000bu:
	case 0xc0e7000cu:
	case 0xc0e7000du:
	case 0xc0e7000eu:
	case 0xc0e7000fu:
	case 0xc0e70010u:
	case 0xc0e70011u:
	case 0xc0e70012u:
	case 0xc0e70013u:
	case 0xc0e70014u:
	case 0xc0e70015u:
	case 0xc0e70016u:
	case 0xc0e70017u:
	case 0xc0e70018u:
	case 0xc0e70019u:
	case 0xc0e7001au:
	case 0xc0e7001bu:
	case 0xc0e7001cu:
	case 0xc0e7001du:
	case 0xc0e7001eu:
	case 0xc0e7001fu:
	case 0xc0e70020u:
	case 0xc0e70021u:
	case 0xc0e70022u:
	case 0xc0e70023u:
	case 0xc0e70024u:
	case 0xc0e70025u:
	case 0xc0e70026u:
	case 0xc0e70027u:
	case 0xc0e80000u:
	case 0xc0e90001u:
	case 0xc0e90002u:
	case 0xc0e90003u:
	case 0xc0e90004u:
	case 0xc0e90005u:
	case 0xc0e90006u:
	case 0xc0e90007u:
	case 0xc0e90008u:
	case 0xc0e90009u:
	case 0xc0e9000au:
	case 0xc0e9000bu:
	case 0xc0e9000cu:
	case 0xc0e9000du:
	case 0xc0ea0001u:
	case 0xc0ea0002u:
	case 0xc0ea0003u:
	case 0xc0ea0004u:
	case 0xc0ea0005u:
	case 0xc0ea0006u:
	case 0xc0ea0007u:
	case 0xc0ea0008u:
	case 0xc0ea0009u:
	case 0xc0ea000au:
	case 0xc0eb0001u:
	case 0xc0eb0002u:
	case 0xc0eb0003u:
	case 0xc0eb0004u:
	case 0xc0eb0005u:
	case 0xc0eb0006u:
	case 0xc0eb0007u:
	case 0xc0ec0000u:
	case 0xc0ec0001u:
	case 0xc0ec0002u:
	case 0xc0ec0003u:
	case 0xc0ec0004u:
	case 0xc0ec0005u:
	case 0xc0ec0006u:
	case 0xc0ec0007u:
	case 0xc0ec0008u:
	case 0xc0ec0009u:
	case 0xc0ec000au:
	case 0xc0ec000bu:
	case 0xc0ec000cu:
		return __tsc(u8"");
// clang-format on
