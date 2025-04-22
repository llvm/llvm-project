// RUN: %clang_cc1 "-triple" "nvptx64-nvidia-cuda" "-target-feature" "+ptx80" "-target-cpu" "sm_90" -emit-llvm -fcuda-is-device -o - %s | FileCheck %s

// CHECK: define{{.*}} void @_Z6kernelPlPvj(
__attribute__((global)) void kernel(long *out, void *ptr, unsigned u) {
  int i = 0;
  // CHECK: call i1 @llvm.nvvm.isspacep.shared.cluster
  out[i++] = __nvvm_isspacep_shared_cluster(ptr);

  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.clusterid.x()
  out[i++] = __nvvm_read_ptx_sreg_clusterid_x();
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.clusterid.y()
  out[i++] = __nvvm_read_ptx_sreg_clusterid_y();
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.clusterid.z()
  out[i++] = __nvvm_read_ptx_sreg_clusterid_z();
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.clusterid.w()
  out[i++] = __nvvm_read_ptx_sreg_clusterid_w();
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nclusterid.x()
  out[i++] = __nvvm_read_ptx_sreg_nclusterid_x();
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nclusterid.y()
  out[i++] = __nvvm_read_ptx_sreg_nclusterid_y();
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nclusterid.z()
  out[i++] = __nvvm_read_ptx_sreg_nclusterid_z();
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.nclusterid.w()
  out[i++] = __nvvm_read_ptx_sreg_nclusterid_w();

  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.cluster.ctaid.x()
  out[i++] = __nvvm_read_ptx_sreg_cluster_ctaid_x();
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.cluster.ctaid.y()
  out[i++] = __nvvm_read_ptx_sreg_cluster_ctaid_y();
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.cluster.ctaid.z()
  out[i++] = __nvvm_read_ptx_sreg_cluster_ctaid_z();
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.cluster.ctaid.w()
  out[i++] = __nvvm_read_ptx_sreg_cluster_ctaid_w();
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.cluster.nctaid.x()
  out[i++] = __nvvm_read_ptx_sreg_cluster_nctaid_x();
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.cluster.nctaid.y()
  out[i++] = __nvvm_read_ptx_sreg_cluster_nctaid_y();
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.cluster.nctaid.z()
  out[i++] = __nvvm_read_ptx_sreg_cluster_nctaid_z();
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.cluster.nctaid.w()
  out[i++] = __nvvm_read_ptx_sreg_cluster_nctaid_w();

  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.cluster.ctarank()
  out[i++] = __nvvm_read_ptx_sreg_cluster_ctarank();
  // CHECK: call i32 @llvm.nvvm.read.ptx.sreg.cluster.nctarank()
  out[i++] = __nvvm_read_ptx_sreg_cluster_nctarank();
  // CHECK: call i1 @llvm.nvvm.is_explicit_cluster()
  out[i++] = __nvvm_is_explicit_cluster();

  auto * sptr = (__attribute__((address_space(3))) void *)ptr;
  // CHECK: call ptr @llvm.nvvm.mapa(ptr %{{.*}}, i32 %{{.*}})
  out[i++] = (long) __nvvm_mapa(ptr, u);
  // CHECK: call ptr addrspace(3) @llvm.nvvm.mapa.shared.cluster(ptr addrspace(3) %{{.*}}, i32 %{{.*}})
  out[i++] = (long) __nvvm_mapa_shared_cluster(sptr, u);
  // CHECK: call i32 @llvm.nvvm.getctarank(ptr {{.*}})
  out[i++] = __nvvm_getctarank(ptr);
  // CHECK: call i32 @llvm.nvvm.getctarank.shared.cluster(ptr addrspace(3) {{.*}})
  out[i++] = __nvvm_getctarank_shared_cluster(sptr);

  // CHECK: call void @llvm.nvvm.barrier.cluster.arrive()
  __nvvm_barrier_cluster_arrive();
  // CHECK: call void @llvm.nvvm.barrier.cluster.arrive.relaxed()
  __nvvm_barrier_cluster_arrive_relaxed();
  // CHECK: call void @llvm.nvvm.barrier.cluster.wait()
  __nvvm_barrier_cluster_wait();
  // CHECK: call void @llvm.nvvm.fence.sc.cluster()
  __nvvm_fence_sc_cluster();

  // CHECK: ret void
}
