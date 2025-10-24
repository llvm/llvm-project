# LLVM fork to work on RISC-V V-ext vectorizing issues

## Repository layout
TODO

## Getting dev shell
* Install `nix`:
```
sh <(curl --proto '=https' --tlsv1.2 -L https://nixos.org/nix/install) --no-daemon
```
* Enable flakes experimental feature:
```
mkdir ~/.config/nix/
echo "experimental-features = nix-command flakes" > ~/.config/nix/nix.conf
```
* Enter to nix shell
```
nix develop
```
* (Optional) Setup [direnv](https://github.com/direnv/direnv) to enter into nix shell automatically


## Building

Inside nix shell:
```
cmake $cmakeFlags -S llvm -B build
ninja -C build
```
