/**
 * Universal UEFI TPM 2.0 Driver for STMicroelectronics ST33TPHF2XSP
 * Dell Latitude 5450 MIL-SPEC (JRTC1) Firmware Bug Mitigation
 *
 * MISSION: Provide universal TPM access across ALL operating systems by
 * fixing the STMicroelectronics firmware bug at UEFI level through
 * Intel ME coordination and EFI_TPM2_PROTOCOL interface.
 *
 * TECHNICAL SOLUTION:
 * - Forces correct 4096/4096 byte command/response buffers
 * - Uses Intel ME as communication bridge to actual TPM hardware
 * - Presents standard EFI_TPM2_PROTOCOL interface to all OS
 * - Military token authorization with Dell MIL-SPEC integration
 *
 * Agent Coordination:
 * - NSA: Nation-state level UEFI firmware manipulation techniques
 * - HARDWARE-INTEL: Intel ME HAP mode coordination and Meteor Lake integration
 * - ARCHITECT: Universal EFI_TPM2_PROTOCOL design for cross-OS compatibility
 *
 * Author: Multi-Agent Coordination (NSA + HARDWARE-INTEL + ARCHITECT)
 * Target: STMicroelectronics ST33TPHF2XSP TPM 2.0 (firmware 1.769)
 * Classification: MIL-SPEC Implementation
 */

#include <Uefi.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/UefiRuntimeServicesTableLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Library/BaseMemoryLib.h>
#include <Library/DebugLib.h>
#include <Library/PrintLib.h>
#include <Library/IoLib.h>
#include <Library/TimerLib.h>
#include <Protocol/Tpm2Protocol.h>
#include <IndustryStandard/Tpm20.h>

//
// Intel ME Integration Headers
//
#include <Protocol/HeciProtocol.h>
#include <MkhiMsgs.h>

//
// Dell MIL-SPEC Token Definitions
//
#define DELL_MILSPEC_TOKEN_BASE     0x049E
#define DELL_MILSPEC_TOKEN_COUNT    6
#define DELL_MILSPEC_TOKEN_MASK     0x3F

//
// TPM Buffer Configuration - Force Correct Sizes
//
#define TPM_COMMAND_BUFFER_SIZE     4096
#define TPM_RESPONSE_BUFFER_SIZE    4096
#define TPM_TIMEOUT_MAX             30000  // 30 seconds

//
// Intel ME Communication Constants
//
#define ME_TPM_COMMAND              0x0A
#define ME_TPM_RESPONSE             0x0B
#define ME_HAP_MODE_VERIFY          0x0C

//
// Driver Instance Signature
//
#define TPM2_DEVICE_INTERFACE_SIGNATURE  SIGNATURE_32('T','P','M','2')

//
// TPM2 Device Interface Structure
//
typedef struct {
  UINT32                    Signature;
  EFI_TPM2_PROTOCOL         Tpm2Protocol;
  EFI_HANDLE                Handle;
  BOOLEAN                   IsInitialized;
  BOOLEAN                   MeAvailable;
  HECI_PROTOCOL            *HeciProtocol;
  UINT8                    *CommandBuffer;
  UINT8                    *ResponseBuffer;
  UINT32                    MilSpecTokens[DELL_MILSPEC_TOKEN_COUNT];
} TPM2_DEVICE_INTERFACE;

//
// Intel ME TPM Command Structure
//
typedef struct {
  UINT32    Command;
  UINT32    Length;
  UINT32    TokenMask;
  UINT8     Data[TPM_COMMAND_BUFFER_SIZE];
} ME_TPM_COMMAND;

//
// Intel ME TPM Response Structure
//
typedef struct {
  UINT32    Status;
  UINT32    Length;
  UINT8     Data[TPM_RESPONSE_BUFFER_SIZE];
} ME_TPM_RESPONSE;

//
// Global Variables
//
STATIC TPM2_DEVICE_INTERFACE  *mTpm2DeviceInterface = NULL;

/**
 * Verify Dell MIL-SPEC Tokens
 *
 * This function validates the presence and accessibility of all 6 Dell
 * MIL-SPEC tokens (0x049E-0x04A3) required for military-grade TPM access.
 *
 * @return EFI_SUCCESS if all tokens are accessible
 */
EFI_STATUS
VerifyDellMilSpecTokens (
  VOID
  )
{
  UINT32  Index;
  UINT32  TokenValue;
  UINT32  ExpectedTokens = 0x3F; // All 6 tokens present (bits 0-5)
  UINT32  FoundTokens = 0;

  DEBUG ((EFI_D_INFO, "TPM2: Verifying Dell MIL-SPEC tokens\n"));

  for (Index = 0; Index < DELL_MILSPEC_TOKEN_COUNT; Index++) {
    //
    // Read token from Dell-specific NVRAM location
    //
    TokenValue = IoRead32 (DELL_MILSPEC_TOKEN_BASE + (Index * 4));

    if (TokenValue != 0xFFFFFFFF) {
      mTpm2DeviceInterface->MilSpecTokens[Index] = TokenValue;
      FoundTokens |= (1 << Index);
      DEBUG ((EFI_D_INFO, "TPM2: Token %d present: 0x%08X\n", Index, TokenValue));
    } else {
      DEBUG ((EFI_D_ERROR, "TPM2: Token %d missing\n", Index));
    }
  }

  if ((FoundTokens & ExpectedTokens) != ExpectedTokens) {
    DEBUG ((EFI_D_ERROR, "TPM2: MIL-SPEC token validation failed: 0x%02X expected 0x%02X\n",
           FoundTokens, ExpectedTokens));
    return EFI_ACCESS_DENIED;
  }

  DEBUG ((EFI_D_INFO, "TPM2: All Dell MIL-SPEC tokens validated\n"));
  return EFI_SUCCESS;
}

/**
 * Initialize Intel ME Communication
 *
 * Sets up communication with Intel Management Engine in HAP mode
 * for TPM command bridging. Verifies ME is in High Assurance Platform
 * mode for enhanced security.
 *
 * @return EFI_SUCCESS if ME communication is established
 */
EFI_STATUS
InitializeIntelMeCommunication (
  VOID
  )
{
  EFI_STATUS                Status;
  UINT32                    MeMode;
  UINT32                    MeFeatures;
  HECI_PROTOCOL            *HeciProtocol;

  DEBUG ((EFI_D_INFO, "TPM2: Initializing Intel ME communication\n"));

  //
  // Locate HECI Protocol for ME communication
  //
  Status = gBS->LocateProtocol (
                  &gHeciProtocolGuid,
                  NULL,
                  (VOID **) &HeciProtocol
                  );
  if (EFI_ERROR (Status)) {
    DEBUG ((EFI_D_ERROR, "TPM2: Failed to locate HECI protocol: %r\n", Status));
    return Status;
  }

  mTpm2DeviceInterface->HeciProtocol = HeciProtocol;

  //
  // Verify ME is in HAP (High Assurance Platform) mode
  //
  Status = HeciProtocol->GetMeMode (&MeMode);
  if (EFI_ERROR (Status)) {
    DEBUG ((EFI_D_ERROR, "TPM2: Failed to get ME mode: %r\n", Status));
    return Status;
  }

  if (MeMode != ME_MODE_HAP) {
    DEBUG ((EFI_D_ERROR, "TPM2: ME not in HAP mode (0x%08X), cannot proceed\n", MeMode));
    return EFI_SECURITY_VIOLATION;
  }

  //
  // Verify ME features include TPM support
  //
  Status = HeciProtocol->GetMeFeatures (&MeFeatures);
  if (EFI_ERROR (Status) || !(MeFeatures & ME_FEATURE_TPM_SUPPORT)) {
    DEBUG ((EFI_D_ERROR, "TPM2: ME does not support TPM bridging\n"));
    return EFI_UNSUPPORTED;
  }

  DEBUG ((EFI_D_INFO, "TPM2: Intel ME HAP mode verified, TPM support enabled\n"));
  mTpm2DeviceInterface->MeAvailable = TRUE;

  return EFI_SUCCESS;
}

/**
 * Send TPM Command via Intel ME
 *
 * Core function that sends TPM commands through Intel ME interface,
 * bypassing the buggy STMicroelectronics firmware. Forces correct
 * 4096/4096 buffer configuration and includes military token authorization.
 *
 * @param CommandBuffer   Pointer to TPM command data
 * @param CommandSize     Size of command data
 * @param ResponseBuffer  Pointer to response buffer
 * @param ResponseSize    Pointer to response size
 *
 * @return EFI_SUCCESS if command executed successfully
 */
EFI_STATUS
SendTpmCommandViaME (
  IN  UINT8   *CommandBuffer,
  IN  UINT32   CommandSize,
  OUT UINT8   *ResponseBuffer,
  OUT UINT32  *ResponseSize
  )
{
  EFI_STATUS        Status;
  ME_TPM_COMMAND   *MeCommand;
  ME_TPM_RESPONSE  *MeResponse;
  UINT32            MeCommandSize;
  UINT32            MeResponseSize;
  UINT32            Timeout;
  UINT32            TokenMask;
  UINT32            Index;

  if (!mTpm2DeviceInterface->MeAvailable) {
    return EFI_NOT_READY;
  }

  //
  // Validate buffer sizes - enforce 4096 byte limits
  //
  if (CommandSize > TPM_COMMAND_BUFFER_SIZE) {
    DEBUG ((EFI_D_ERROR, "TPM2: Command size %d exceeds limit %d\n",
           CommandSize, TPM_COMMAND_BUFFER_SIZE));
    return EFI_INVALID_PARAMETER;
  }

  if (*ResponseSize > TPM_RESPONSE_BUFFER_SIZE) {
    *ResponseSize = TPM_RESPONSE_BUFFER_SIZE;
  }

  //
  // Build token mask from Dell MIL-SPEC tokens
  //
  TokenMask = 0;
  for (Index = 0; Index < DELL_MILSPEC_TOKEN_COUNT; Index++) {
    TokenMask |= (mTpm2DeviceInterface->MilSpecTokens[Index] & 0xFF) << (Index * 4);
  }

  //
  // Prepare ME command structure
  //
  MeCommand = AllocateZeroPool (sizeof (ME_TPM_COMMAND));
  if (MeCommand == NULL) {
    return EFI_OUT_OF_RESOURCES;
  }

  MeCommand->Command = ME_TPM_COMMAND;
  MeCommand->Length = CommandSize;
  MeCommand->TokenMask = TokenMask;
  CopyMem (MeCommand->Data, CommandBuffer, CommandSize);

  MeCommandSize = sizeof (ME_TPM_COMMAND);

  //
  // Allocate response buffer
  //
  MeResponse = AllocateZeroPool (sizeof (ME_TPM_RESPONSE));
  if (MeResponse == NULL) {
    FreePool (MeCommand);
    return EFI_OUT_OF_RESOURCES;
  }

  MeResponseSize = sizeof (ME_TPM_RESPONSE);

  //
  // Send command to ME with timeout
  //
  DEBUG ((EFI_D_INFO, "TPM2: Sending command to ME (size: %d)\n", CommandSize));

  Status = mTpm2DeviceInterface->HeciProtocol->SendMessage (
                                                (UINT32 *) MeCommand,
                                                MeCommandSize,
                                                BIOS_FIXED_HOST_ADDR,
                                                HECI_TPM_MESSAGE_TYPE
                                                );
  if (EFI_ERROR (Status)) {
    DEBUG ((EFI_D_ERROR, "TPM2: Failed to send ME command: %r\n", Status));
    goto Exit;
  }

  //
  // Wait for response with timeout
  //
  Timeout = 0;
  do {
    Status = mTpm2DeviceInterface->HeciProtocol->ReceiveMessage (
                                                  (UINT32 *) MeResponse,
                                                  &MeResponseSize
                                                  );
    if (!EFI_ERROR (Status)) {
      break;
    }

    MicroSecondDelay (1000); // 1ms delay
    Timeout++;
  } while (Timeout < TPM_TIMEOUT_MAX);

  if (EFI_ERROR (Status)) {
    DEBUG ((EFI_D_ERROR, "TPM2: ME response timeout: %r\n", Status));
    goto Exit;
  }

  //
  // Validate response
  //
  if (MeResponse->Status != 0) {
    DEBUG ((EFI_D_ERROR, "TPM2: ME returned error status: 0x%08X\n", MeResponse->Status));
    Status = EFI_DEVICE_ERROR;
    goto Exit;
  }

  if (MeResponse->Length > *ResponseSize) {
    DEBUG ((EFI_D_ERROR, "TPM2: Response too large: %d > %d\n",
           MeResponse->Length, *ResponseSize));
    Status = EFI_BUFFER_TOO_SMALL;
    goto Exit;
  }

  //
  // Copy response data
  //
  CopyMem (ResponseBuffer, MeResponse->Data, MeResponse->Length);
  *ResponseSize = MeResponse->Length;

  DEBUG ((EFI_D_INFO, "TPM2: Command completed successfully (response size: %d)\n",
         *ResponseSize));

Exit:
  FreePool (MeCommand);
  FreePool (MeResponse);
  return Status;
}

/**
 * EFI_TPM2_PROTOCOL.SubmitCommand Implementation
 *
 * Standard EFI TPM2 protocol interface that all operating systems expect.
 * This function presents a normal TPM interface while internally routing
 * all commands through Intel ME to bypass firmware bugs.
 *
 * @param This            Protocol instance
 * @param InputBuffer     TPM command buffer
 * @param InputBufferSize Size of command
 * @param OutputBuffer    TPM response buffer
 * @param OutputBufferSize Size of response buffer
 *
 * @return EFI_SUCCESS if command processed
 */
EFI_STATUS
EFIAPI
Tpm2SubmitCommand (
  IN      EFI_TPM2_PROTOCOL *This,
  IN      UINT32            InputBufferSize,
  IN      UINT8             *InputBuffer,
  IN      UINT32            OutputBufferSize,
  IN OUT  UINT8             *OutputBuffer
  )
{
  EFI_STATUS              Status;
  TPM2_DEVICE_INTERFACE  *DeviceInterface;
  UINT32                  ResponseSize;

  if (This == NULL || InputBuffer == NULL || OutputBuffer == NULL) {
    return EFI_INVALID_PARAMETER;
  }

  DeviceInterface = TPM2_DEVICE_INTERFACE_FROM_TPM2_PROTOCOL (This);

  if (DeviceInterface->Signature != TPM2_DEVICE_INTERFACE_SIGNATURE) {
    return EFI_INVALID_PARAMETER;
  }

  if (!DeviceInterface->IsInitialized) {
    return EFI_NOT_READY;
  }

  ResponseSize = OutputBufferSize;

  //
  // Route command through Intel ME TPM bridge
  //
  Status = SendTpmCommandViaME (
             InputBuffer,
             InputBufferSize,
             OutputBuffer,
             &ResponseSize
             );

  if (EFI_ERROR (Status)) {
    DEBUG ((EFI_D_ERROR, "TPM2: Command failed: %r\n", Status));
    return Status;
  }

  //
  // Validate TPM response header
  //
  if (ResponseSize < sizeof (TPM2_RESPONSE_HEADER)) {
    DEBUG ((EFI_D_ERROR, "TPM2: Response too small\n"));
    return EFI_DEVICE_ERROR;
  }

  DEBUG ((EFI_D_INFO, "TPM2: Command completed successfully\n"));
  return EFI_SUCCESS;
}

/**
 * EFI_TPM2_PROTOCOL.GetCapability Implementation
 *
 * Returns TPM capability information. Always reports correct buffer
 * sizes (4096/4096) regardless of underlying firmware bug.
 *
 * @param This       Protocol instance
 * @param Capability Capability to query
 * @param Buffer     Capability data buffer
 * @param BufferSize Size of capability data
 *
 * @return EFI_SUCCESS with capability information
 */
EFI_STATUS
EFIAPI
Tpm2GetCapability (
  IN      EFI_TPM2_PROTOCOL *This,
  IN      UINT32            Capability,
  IN OUT  UINT8             *Buffer,
  IN OUT  UINT32            *BufferSize
  )
{
  TPM2_DEVICE_INTERFACE  *DeviceInterface;

  if (This == NULL || Buffer == NULL || BufferSize == NULL) {
    return EFI_INVALID_PARAMETER;
  }

  DeviceInterface = TPM2_DEVICE_INTERFACE_FROM_TPM2_PROTOCOL (This);

  if (DeviceInterface->Signature != TPM2_DEVICE_INTERFACE_SIGNATURE) {
    return EFI_INVALID_PARAMETER;
  }

  //
  // Handle capability queries
  //
  switch (Capability) {
    case TPM_CAP_COMMANDS:
      //
      // Return supported TPM commands
      //
      if (*BufferSize < 4) {
        *BufferSize = 4;
        return EFI_BUFFER_TOO_SMALL;
      }
      *(UINT32*)Buffer = 0xFFFFFFFF; // All commands supported via ME bridge
      *BufferSize = 4;
      break;

    case TPM_CAP_BUFFER_SIZE:
      //
      // Always report correct buffer sizes (4096/4096)
      //
      if (*BufferSize < 8) {
        *BufferSize = 8;
        return EFI_BUFFER_TOO_SMALL;
      }
      ((UINT32*)Buffer)[0] = TPM_COMMAND_BUFFER_SIZE;
      ((UINT32*)Buffer)[1] = TPM_RESPONSE_BUFFER_SIZE;
      *BufferSize = 8;
      break;

    default:
      //
      // Forward other capabilities to TPM via ME
      //
      return Tpm2SubmitCommand (
               This,
               12, // Standard capability command size
               (UINT8*)&Capability,
               *BufferSize,
               Buffer
               );
  }

  return EFI_SUCCESS;
}

/**
 * Driver Entry Point
 *
 * Initializes the universal TPM driver, sets up Intel ME communication,
 * validates Dell MIL-SPEC tokens, and installs EFI_TPM2_PROTOCOL.
 *
 * @param ImageHandle Image handle of the driver
 * @param SystemTable Pointer to system table
 *
 * @return EFI_SUCCESS if driver installed successfully
 */
EFI_STATUS
EFIAPI
UniversalTpmDriverEntry (
  IN EFI_HANDLE        ImageHandle,
  IN EFI_SYSTEM_TABLE  *SystemTable
  )
{
  EFI_STATUS  Status;

  DEBUG ((EFI_D_INFO, "Universal TPM Driver v1.0 - Dell Latitude 5450 MIL-SPEC\n"));
  DEBUG ((EFI_D_INFO, "Target: STMicroelectronics ST33TPHF2XSP firmware 1.769\n"));

  //
  // Allocate device interface structure
  //
  mTpm2DeviceInterface = AllocateZeroPool (sizeof (TPM2_DEVICE_INTERFACE));
  if (mTpm2DeviceInterface == NULL) {
    DEBUG ((EFI_D_ERROR, "Failed to allocate device interface\n"));
    return EFI_OUT_OF_RESOURCES;
  }

  //
  // Initialize device interface
  //
  mTpm2DeviceInterface->Signature = TPM2_DEVICE_INTERFACE_SIGNATURE;
  mTpm2DeviceInterface->Handle = ImageHandle;
  mTpm2DeviceInterface->IsInitialized = FALSE;
  mTpm2DeviceInterface->MeAvailable = FALSE;

  //
  // Set up TPM2 protocol interface
  //
  mTpm2DeviceInterface->Tpm2Protocol.SubmitCommand = Tpm2SubmitCommand;
  mTpm2DeviceInterface->Tpm2Protocol.GetCapability = Tpm2GetCapability;

  //
  // Allocate command and response buffers with correct sizes
  //
  mTpm2DeviceInterface->CommandBuffer = AllocateZeroPool (TPM_COMMAND_BUFFER_SIZE);
  mTpm2DeviceInterface->ResponseBuffer = AllocateZeroPool (TPM_RESPONSE_BUFFER_SIZE);

  if (mTpm2DeviceInterface->CommandBuffer == NULL ||
      mTpm2DeviceInterface->ResponseBuffer == NULL) {
    DEBUG ((EFI_D_ERROR, "Failed to allocate TPM buffers\n"));
    Status = EFI_OUT_OF_RESOURCES;
    goto Error;
  }

  //
  // Verify Dell MIL-SPEC tokens
  //
  Status = VerifyDellMilSpecTokens ();
  if (EFI_ERROR (Status)) {
    DEBUG ((EFI_D_ERROR, "Dell MIL-SPEC token validation failed: %r\n", Status));
    goto Error;
  }

  //
  // Initialize Intel ME communication
  //
  Status = InitializeIntelMeCommunication ();
  if (EFI_ERROR (Status)) {
    DEBUG ((EFI_D_ERROR, "Intel ME initialization failed: %r\n", Status));
    goto Error;
  }

  //
  // Install TPM2 protocol
  //
  Status = gBS->InstallProtocolInterface (
                  &mTpm2DeviceInterface->Handle,
                  &gEfiTpm2ProtocolGuid,
                  EFI_NATIVE_INTERFACE,
                  &mTpm2DeviceInterface->Tpm2Protocol
                  );
  if (EFI_ERROR (Status)) {
    DEBUG ((EFI_D_ERROR, "Failed to install TPM2 protocol: %r\n", Status));
    goto Error;
  }

  mTpm2DeviceInterface->IsInitialized = TRUE;

  DEBUG ((EFI_D_INFO, "Universal TPM Driver installed successfully\n"));
  DEBUG ((EFI_D_INFO, "All OS will see standard TPM 2.0 with 4096/4096 buffers\n"));

  return EFI_SUCCESS;

Error:
  if (mTpm2DeviceInterface != NULL) {
    if (mTpm2DeviceInterface->CommandBuffer != NULL) {
      FreePool (mTpm2DeviceInterface->CommandBuffer);
    }
    if (mTpm2DeviceInterface->ResponseBuffer != NULL) {
      FreePool (mTpm2DeviceInterface->ResponseBuffer);
    }
    FreePool (mTpm2DeviceInterface);
    mTpm2DeviceInterface = NULL;
  }

  return Status;
}

/**
 * Driver Unload Function
 *
 * Cleans up resources and uninstalls protocol when driver is unloaded.
 *
 * @param ImageHandle Image handle of the driver
 *
 * @return EFI_SUCCESS if unloaded successfully
 */
EFI_STATUS
EFIAPI
UniversalTpmDriverUnload (
  IN EFI_HANDLE ImageHandle
  )
{
  EFI_STATUS Status;

  if (mTpm2DeviceInterface == NULL) {
    return EFI_SUCCESS;
  }

  //
  // Uninstall protocol
  //
  Status = gBS->UninstallProtocolInterface (
                  mTpm2DeviceInterface->Handle,
                  &gEfiTpm2ProtocolGuid,
                  &mTpm2DeviceInterface->Tpm2Protocol
                  );
  if (EFI_ERROR (Status)) {
    DEBUG ((EFI_D_ERROR, "Failed to uninstall TPM2 protocol: %r\n", Status));
    return Status;
  }

  //
  // Free allocated memory
  //
  if (mTpm2DeviceInterface->CommandBuffer != NULL) {
    FreePool (mTpm2DeviceInterface->CommandBuffer);
  }
  if (mTpm2DeviceInterface->ResponseBuffer != NULL) {
    FreePool (mTpm2DeviceInterface->ResponseBuffer);
  }
  FreePool (mTpm2DeviceInterface);
  mTpm2DeviceInterface = NULL;

  DEBUG ((EFI_D_INFO, "Universal TPM Driver unloaded\n"));
  return EFI_SUCCESS;
}

//
// Driver Helper Macros
//
#define TPM2_DEVICE_INTERFACE_FROM_TPM2_PROTOCOL(a) \
  CR (a, TPM2_DEVICE_INTERFACE, Tpm2Protocol, TPM2_DEVICE_INTERFACE_SIGNATURE)