#!/bin/bash
# Add ALL Intel NPU, GNA, DSMIL, and AI acceleration flags to running build

cd ~/kernel-build/linux-source-6.16

echo "Adding comprehensive Intel + DSMIL support..."

# Stop current build to reconfigure
killall make 2>/dev/null
sleep 2

#═══════════════════════════════════════════════════════════════════════════════
# INTEL NPU (Neural Processing Unit) - Critical!
#═══════════════════════════════════════════════════════════════════════════════
scripts/config --module CONFIG_INTEL_MEI
scripts/config --module CONFIG_INTEL_MEI_ME
scripts/config --module CONFIG_INTEL_MEI_TXE
scripts/config --module CONFIG_INTEL_MEI_GSC
scripts/config --module CONFIG_INTEL_MEI_HDCP
scripts/config --module CONFIG_INTEL_MEI_PXP
scripts/config --enable CONFIG_AUXILIARY_BUS
scripts/config --enable CONFIG_VFIO_MDEV
scripts/config --enable CONFIG_VFIO_MDEV_DEVICE

#═══════════════════════════════════════════════════════════════════════════════
# INTEL GNA (Gaussian Neural Accelerator) - Audio DSP with AI
#═══════════════════════════════════════════════════════════════════════════════
scripts/config --enable CONFIG_SND_SOC
scripts/config --enable CONFIG_SND_SOC_INTEL_SST_TOPLEVEL
scripts/config --module CONFIG_SND_SOC_INTEL_CATPT
scripts/config --enable CONFIG_SND_SOC_INTEL_MACH
scripts/config --module CONFIG_SND_SOC_INTEL_BYT_RT5640_MACH
scripts/config --module CONFIG_SND_SOC_INTEL_BYT_MAX98090_MACH
scripts/config --module CONFIG_SND_SOC_INTEL_BDW_RT5650_MACH
scripts/config --module CONFIG_SND_SOC_INTEL_BDW_RT5677_MACH
scripts/config --module CONFIG_SND_SOC_INTEL_BROADWELL_MACH
scripts/config --module CONFIG_SND_SOC_INTEL_BYTCR_RT5640_MACH
scripts/config --module CONFIG_SND_SOC_INTEL_BYTCR_RT5651_MACH
scripts/config --module CONFIG_SND_SOC_INTEL_BYTCR_WM5102_MACH
scripts/config --module CONFIG_SND_SOC_INTEL_CHT_BSW_RT5672_MACH
scripts/config --module CONFIG_SND_SOC_INTEL_CHT_BSW_RT5645_MACH
scripts/config --module CONFIG_SND_SOC_INTEL_CHT_BSW_MAX98090_TI_MACH
scripts/config --module CONFIG_SND_SOC_INTEL_CHT_BSW_NAU8824_MACH
scripts/config --module CONFIG_SND_SOC_INTEL_BYT_CHT_CX2072X_MACH
scripts/config --module CONFIG_SND_SOC_INTEL_BYT_CHT_DA7213_MACH
scripts/config --module CONFIG_SND_SOC_INTEL_BYT_CHT_ES8316_MACH
scripts/config --module CONFIG_SND_SOC_INTEL_BYT_CHT_NOCODEC_MACH
scripts/config --enable CONFIG_SND_SOC_SOF_TOPLEVEL
scripts/config --module CONFIG_SND_SOC_SOF_PCI
scripts/config --module CONFIG_SND_SOC_SOF_ACPI
scripts/config --module CONFIG_SND_SOC_SOF_INTEL_TOPLEVEL
scripts/config --module CONFIG_SND_SOC_SOF_INTEL_HداDA_COMMON
scripts/config --module CONFIG_SND_SOC_SOF_INTEL_COMMON
scripts/config --module CONFIG_SND_SOC_SOF_METEORLAKE

#═══════════════════════════════════════════════════════════════════════════════
# INTEL ARC GPU (i915) - Full Configuration
#═══════════════════════════════════════════════════════════════════════════════
scripts/config --module CONFIG_DRM_I915
scripts/config --enable CONFIG_DRM_I915_FORCE_PROBE --set-str "7d55,7d60,7d45"
scripts/config --enable CONFIG_DRM_I915_USERPTR
scripts/config --enable CONFIG_DRM_I915_GVT
scripts/config --enable CONFIG_DRM_I915_GVT_KVMGT
scripts/config --enable CONFIG_DRM_I915_CAPTURE_ERROR
scripts/config --enable CONFIG_DRM_I915_COMPRESS_ERROR
scripts/config --enable CONFIG_DRM_I915_DEBUG
scripts/config --enable CONFIG_DRM_I915_DEBUG_GEM
scripts/config --enable CONFIG_DRM_I915_ERRLOG_GEM
scripts/config --enable CONFIG_DRM_I915_TRACE_GEM
scripts/config --enable CONFIG_DRM_I915_TRACE_GTT
scripts/config --enable CONFIG_DRM_I915_SW_FENCE_DEBUG_OBJECTS
scripts/config --enable CONFIG_DRM_I915_SW_FENCE_CHECK_DAG
scripts/config --enable CONFIG_DRM_I915_TIMESLICE_DURATION --set-val 1

#═══════════════════════════════════════════════════════════════════════════════
# INTEL SPECIFIC CPU FEATURES
#═══════════════════════════════════════════════════════════════════════════════
scripts/config --enable CONFIG_X86_INTEL_LPSS
scripts/config --enable CONFIG_INTEL_IDLE
scripts/config --enable CONFIG_INTEL_PSTATE
scripts/config --enable CONFIG_X86_INTEL_PSTATE_HWP_CPUFREQ
scripts/config --enable CONFIG_INTEL_UNCORE_FREQ_CONTROL
scripts/config --enable CONFIG_INTEL_HFI_THERMAL
scripts/config --enable CONFIG_INTEL_TCC_COOLING
scripts/config --enable CONFIG_INTEL_POWERCLAMP
scripts/config --enable CONFIG_INTEL_SOC_DTS_THERMAL
scripts/config --enable CONFIG_INTEL_PCH_THERMAL
scripts/config --enable CONFIG_INTEL_TH
scripts/config --module CONFIG_INTEL_TH_PCI
scripts/config --module CONFIG_INTEL_TH_ACPI
scripts/config --module CONFIG_INTEL_TH_GTH
scripts/config --module CONFIG_INTEL_TH_STH
scripts/config --module CONFIG_INTEL_TH_MSU
scripts/config --module CONFIG_INTEL_TH_PTI

#═══════════════════════════════════════════════════════════════════════════════
# AVX-512 and Vector Extensions (AI Workloads)
#═══════════════════════════════════════════════════════════════════════════════
scripts/config --enable CONFIG_X86_AVX512
scripts/config --enable CONFIG_CRYPTO_AVX512
scripts/config --enable CONFIG_CRYPTO_POLY1305_X86_64
scripts/config --enable CONFIG_CRYPTO_NHPOLY1305_AVX2
scripts/config --enable CONFIG_CRYPTO_CURVE25519_X86
scripts/config --enable CONFIG_CRYPTO_BLAKE2S_X86

#═══════════════════════════════════════════════════════════════════════════════
# DSMIL HARDWARE SUPPORT
#═══════════════════════════════════════════════════════════════════════════════
# MSR access (Model Specific Registers - required for DSMIL)
scripts/config --module CONFIG_X86_MSR
scripts/config --module CONFIG_X86_CPUID

# Dell SMBIOS/WMI (DSMIL uses these)
scripts/config --enable CONFIG_ACPI_WMI
scripts/config --module CONFIG_DELL_SMBIOS
scripts/config --module CONFIG_DELL_SMBIOS_WMI
scripts/config --module CONFIG_DELL_SMBIOS_SMM
scripts/config --module CONFIG_DELL_WMI
scripts/config --module CONFIG_DELL_WMI_DESCRIPTOR
scripts/config --module CONFIG_DELL_WMI_AIO
scripts/config --module CONFIG_DELL_WMI_LED
scripts/config --module CONFIG_DELL_LAPTOP
scripts/config --module CONFIG_DELL_RBTN

# ACPI platform (DSMIL device enumeration)
scripts/config --enable CONFIG_ACPI
scripts/config --enable CONFIG_ACPI_AC
scripts/config --enable CONFIG_ACPI_BATTERY
scripts/config --enable CONFIG_ACPI_BUTTON
scripts/config --enable CONFIG_ACPI_VIDEO
scripts/config --enable CONFIG_ACPI_FAN
scripts/config --enable CONFIG_ACPI_DOCK
scripts/config --enable CONFIG_ACPI_PROCESSOR
scripts/config --enable CONFIG_ACPI_THERMAL
scripts/config --enable CONFIG_ACPI_CUSTOM_DSDT
scripts/config --enable CONFIG_ACPI_HOTPLUG_CPU
scripts/config --enable CONFIG_ACPI_HOTPLUG_MEMORY

# PCI access (DSMIL device control)
scripts/config --enable CONFIG_PCI
scripts/config --enable CONFIG_PCIEPORTBUS
scripts/config --enable CONFIG_PCIEASPM
scripts/config --enable CONFIG_PCIE_PTM
scripts/config --enable CONFIG_PCI_MSI
scripts/config --enable CONFIG_PCI_REALLOC_ENABLE_AUTO

#═══════════════════════════════════════════════════════════════════════════════
# INTEL TELEMETRY (Hardware monitoring for DSMIL)
#═══════════════════════════════════════════════════════════════════════════════
scripts/config --module CONFIG_INTEL_TELEMETRY
scripts/config --module CONFIG_INTEL_PMC_CORE
scripts/config --module CONFIG_INTEL_PUNIT_IPC

#═══════════════════════════════════════════════════════════════════════════════
# NCS2 (Neural Compute Stick 2) - USB ML accelerator
#═══════════════════════════════════════════════════════════════════════════════
scripts/config --enable CONFIG_USB
scripts/config --enable CONFIG_USB_XHCI_HCD
scripts/config --enable CONFIG_USB_XHCI_PCI
scripts/config --module CONFIG_USB_STORAGE
scripts/config --module CONFIG_USB_ACM

echo "All Intel + DSMIL flags added!"

# Rebuild with all flags
echo "Restarting build with full configuration..."
yes "" | make -j15 KCFLAGS="-march=alderlake -O2 -pipe -fno-plt -ftree-vectorize" \
    LOCALVERSION=-ultimate olddefconfig bindeb-pkg > ~/ultimate-build-full.log 2>&1 &

echo "BUILD RESTARTED with full Intel/DSMIL support!"
echo "Monitor: tail -f ~/ultimate-build-full.log"
