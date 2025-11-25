#!/bin/bash
# Set proper kernel command line for ZFSBootMenu with Intel GuC and security

echo "1786" | sudo -S zpool import -f rpool
echo "1786" | sudo -S bash -c 'echo "1/0523/600260" | zfs load-key rpool'

# Set comprehensive kernel command line with all Intel + Security params
echo "1786" | sudo -S zfs set org.zfsbootmenu:commandline="intel_iommu=on iommu=pt zfs.zfs_arc_max=25769803776 pti=on mitigations=auto,nosmt init_on_alloc=1 init_on_free=1 early_unicode=1 intel_pstate=active processor.max_cstate=1 intel_idle.max_cstate=0 i915.enable_guc=3 i915.enable_fbc=1 i915.enable_psr=2 intel_npu.enable=1 intel_gna.enable=1 isolcpus=12-15 rcu_nocbs=12-15 clearcpuid=304 tsx=on thunderbolt.security=user module.sig_enforce=1 lockdown=confidentiality spectre_v2=on spec_store_bypass_disable=on l1tf=full,force mds=full" rpool/ROOT/ultimate-xen-ai

echo "Kernel commandline set for ultimate-xen-ai!"

# Verify
echo "1786" | sudo -S zfs get org.zfsbootmenu:commandline rpool/ROOT/ultimate-xen-ai

echo "1786" | sudo -S zpool export rpool

echo ""
echo "READY TO REBOOT with full Intel GuC + Security parameters!"
echo "sudo reboot"
