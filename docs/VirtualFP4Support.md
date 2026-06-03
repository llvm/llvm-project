# 虚拟 FP4/MXFP4 硬件支持实现文档

## 概述
本文档描述了在 AMDGPU 后端中实现虚拟 FP4 和 MXFP4 支持的设计方案。基于 "浑天" 虚拟硬件原理，我们创建了一个软件模拟层，可以在不支持原生 FP4 指令的硬件上实现 FP4 和 MXFP4 操作。

## 设计原理

### 虚拟硬件模型
- 基于 "浑天" 虚拟硬件的设计理念
- 使用现有 INT4 硬件作为基础
- 通过量化/反量化实现 FP4 操作
- 通过块缩放实现 MXFP4 操作

### 数据格式

#### FP4 格式
- 总长度：4 位
- 结构：1 位符号位 + 2 位指数位 + 1 位尾数位
- 表示范围：近似 -7.0 到 +7.0

#### MXFP4 格式
- 数据：4 位整数
- 缩放：8 位块缩放因子 (UE8M0)
- 实现：通过 INT4 硬件 + 块缩放

## 实现组件

### 1. 虚拟硬件层
- `VirtualFp4HwState` - 虚拟硬件状态
- `init_virtual_fp4_hw()` - 初始化虚拟硬件
- 各种 FP4/MXFP4 操作的实现

### 2. LLVM IR 层
- `IntrinsicsVFP4.h` - 定义虚拟指令接口
- 支持 FP4 转换、算术运算和 MXFP4 操作

### 3. AMDGPU 后端层
- `AMDGPUVirtualFP4Lowering` - 将虚拟指令降低为实际操作
- 集成到现有 SWMMAC 框架

## 使用方法

### 编译器层面
```cpp
// 使用虚拟 FP4 操作
%result = call <4 x i4> @llvm.vfp4.add(<4 x i4> %a, <4 x i4> %b, float %scale)
```

### 运行时层面
虚拟硬件会自动处理量化、运算和反量化过程。

## 性能考量

### 优势
- 兼容现有硬件 (gfx1200/RDNA4)
- 可以利用 INT4 硬件加速
- 通过块缩放提高 MXFP4 精度

### 限制
- 性能低于原生 FP4 指令
- 额外的量化/反量化开销
- 需要额外的缩放因子存储

## 未来扩展

1. 优化量化算法
2. 支持更多 FP4 操作
3. 集成到 MLIR 中
4. 优化矩阵乘法实现

## 参考资料

- `VirtualFp4Hw.h` - 虚拟硬件接口定义
- `VirtualFp4Hw.cpp` - 虚拟硬件实现
- `IntrinsicsVFP4.h` - LLVM IR 接口
- `AMDGPUVirtualFP4.cpp` - AMDGPU 后端集成