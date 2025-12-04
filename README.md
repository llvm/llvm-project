# LLVM fork to work on RISC-V V-ext vectorizing issues

## Working on LLVM

### Getting dev shell
* Install `nix`:
```
sh <(curl --proto '=https' --tlsv1.2 -L https://nixos.org/nix/install) --no-daemon
```
* Enable flakes experimental feature:
```
mkdir ~/.config/nix/
echo "experimental-features = nix-command flakes impure-derivations ca-derivations" > ~/.config/nix/nix.conf
```
* Enter to nix shell
```
nix develop '.#toolchain'
```
* (Optional) Setup [direnv](https://github.com/direnv/direnv) to enter into nix shell automatically

### Setup connection to binary cache
#### Why is that necessary?
Every nix package have unique hash, which can be computed before building process.
When you build any thing, nix firstly tries to find built package across connected binary caches.
Cross LLVM build process requires a lot of tools,
not all of which are available as binary package in public package repositories.
After first LLVM build (which I'am already done) all of this packages become available in our binary package cache.

#### Setup ssh access
Firstly, you need setup ssh access to specified user **for `root` user in your system**.
```
host vityaman-nix-storage
    user nix-storage
    hostname 62.84.116.90
    port 22
    identityfile ~/.ssh/nix-storage_id_rsa
```
Private key can be found in telegram developers chat.
#### Setup nix substituters
Next, you need add our host into *substituters*.
Usually, it can be done by adding following line into `~/.config/nix/nix.conf`:
```
substituters = ssh://vityaman-nix-storage
```

### Building LLVM
Enter into nix shell:
```
nix develop
```

Then build libllvm in normal way:
```
cmake $cmakeFlags -S llvm -B build
ninja -C build
```

After that, you can use built libllvm with upstream clang, which preinstalled in nix shell.
Minimal example to compile code for Linux/RISC-V environment.
```
// inside `nix develop` shell
riscv64-unknown-linux-gnu-clang++ hello.cpp -emit-llvm -S -o hello.ll
../build/bin/llc -march=riscv64 --relocation-model=pic hello.ll -o hello.s
riscv64-unknown-linux-gnu-clang++ hello.s hello.o
```
For details see [LLVM docs](https://llvm.org/docs/GettingStarted.html).

### Running performance testing
TODO[khaser]: not implemented yet

### Running regression testing
TODO[khaser]: not implemented yet

### Misc
Original LLVM readme can be found in [./README-llvm.md](./README-llvm.md)

### Infra notes
TODO[khaser]: write notes

