# srtp小组 可用编译命令

1. 设置交换分区
```bash
sudo rm /var/cache/swap/swap0
sudo swapoff -a
sudo mkdir -p /var/cache/swap/
sudo dd if=/dev/zero of=/var/cache/swap/swap0 bs=64M count=256
#（这里的count可以小一点，1024完全ok，更小一些不知道够不够）
sudo chmod 0600 /var/cache/swap/swap0
sudo mkswap /var/cache/swap/swap0
sudo swapon /var/cache/swap/swap0
sudo swapon -s
```

2. 进入build文件夹编译
```bash
cd llvm-project
sudo mkdir build
cd build
```
3. 编译
```bash
sudo cmake -G "Ninja" -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;compiler-rt;" ../llvm -DCMAKE_BUILD_TYPE=release &&sudo ninja -j3
```
ninja命令后的-j后跟的数字可根据自己电脑内核数修改，建议比电脑配置内核数少一个，否则电脑其他进程会很卡


