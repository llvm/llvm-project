#!/usr/bin/env bash

set -e

usage()
{
    echo "Usage: $0 [BuildArch] [CodeName] [lldbx.y] [llvmx[.y]] [--skipunmount] --rootfsdir <directory>]"
    echo "BuildArch can be: arm(default), arm64, armel, armv6, loongarch64, ppc64le, riscv64, s390x, x64, x86"
    echo "CodeName - optional, Code name for Linux, can be: xenial(default), zesty, bionic, alpine"
    echo "                               for alpine can be specified with version: alpineX.YY or alpineedge"
    echo "                               for FreeBSD can be: freebsd13, freebsd14"
    echo "                               for illumos can be: illumos"
    echo "                               for Haiku can be: haiku."
    echo "lldbx.y - optional, LLDB version, can be: lldb3.9(default), lldb4.0, lldb5.0, lldb6.0 no-lldb. Ignored for alpine and FreeBSD"
    echo "llvmx[.y] - optional, LLVM version for LLVM related packages."
    echo "--skipunmount - optional, will skip the unmount of rootfs folder."
    echo "--skipsigcheck - optional, will skip package signature checks (allowing untrusted packages)."
    echo "--skipemulation - optional, will skip qemu and debootstrap requirement when building environment for debian based systems."
    echo "--use-mirror - optional, use mirror URL to fetch resources, when available."
    echo "--jobs N - optional, restrict to N jobs."
    exit 1
}

__CodeName=xenial
__CrossDir=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
__BuildArch=arm
__AlpineArch=armv7
__FreeBSDArch=arm
__FreeBSDMachineArch=armv7
__IllumosArch=arm7
__HaikuArch=arm
__QEMUArch=arm
__UbuntuArch=armhf
__UbuntuRepo=
__UbuntuSuites="updates security backports"
__DebianSuites=
__LLDB_Package="liblldb-3.9-dev"
__SkipUnmount=0

# base development support
__UbuntuPackages="build-essential"

__AlpinePackages="alpine-base"
__AlpinePackages+=" build-base"
__AlpinePackages+=" linux-headers"
__AlpinePackages+=" lldb-dev"
__AlpinePackages+=" python3"
__AlpinePackages+=" libedit"

# symlinks fixer
__UbuntuPackages+=" symlinks"

# runtime dependencies
__UbuntuPackages+=" libicu-dev"
__UbuntuPackages+=" liblttng-ust-dev"
__UbuntuPackages+=" libunwind8-dev"

__AlpinePackages+=" gettext-dev"
__AlpinePackages+=" icu-dev"
__AlpinePackages+=" libunwind-dev"
__AlpinePackages+=" lttng-ust-dev"
__AlpinePackages+=" compiler-rt"

# runtime libraries' dependencies
__UbuntuPackages+=" libcurl4-openssl-dev"
__UbuntuPackages+=" libkrb5-dev"
__UbuntuPackages+=" libssl-dev"
__UbuntuPackages+=" zlib1g-dev"
__UbuntuPackages+=" libbrotli-dev"

__AlpinePackages+=" curl-dev"
__AlpinePackages+=" krb5-dev"
__AlpinePackages+=" openssl-dev"
__AlpinePackages+=" zlib-dev"

__FreeBSDBase="13.4-RELEASE"
__FreeBSDPkg="1.21.3"
__FreeBSDABI="13"
__FreeBSDPackages="libunwind"
__FreeBSDPackages+=" icu"
__FreeBSDPackages+=" libinotify"
__FreeBSDPackages+=" openssl"
__FreeBSDPackages+=" krb5"
__FreeBSDPackages+=" terminfo-db"

__IllumosPackages="icu"
__IllumosPackages+=" mit-krb5"
__IllumosPackages+=" openssl"
__IllumosPackages+=" zlib"

__HaikuPackages="gcc_syslibs"
__HaikuPackages+=" gcc_syslibs_devel"
__HaikuPackages+=" gmp"
__HaikuPackages+=" gmp_devel"
__HaikuPackages+=" icu[0-9]+"
__HaikuPackages+=" icu[0-9]*_devel"
__HaikuPackages+=" krb5"
__HaikuPackages+=" krb5_devel"
__HaikuPackages+=" libiconv"
__HaikuPackages+=" libiconv_devel"
__HaikuPackages+=" llvm[0-9]*_libunwind"
__HaikuPackages+=" llvm[0-9]*_libunwind_devel"
__HaikuPackages+=" mpfr"
__HaikuPackages+=" mpfr_devel"
__HaikuPackages+=" openssl3"
__HaikuPackages+=" openssl3_devel"
__HaikuPackages+=" zlib"
__HaikuPackages+=" zlib_devel"

# ML.NET dependencies
__UbuntuPackages+=" libomp5"
__UbuntuPackages+=" libomp-dev"

# Taken from https://github.com/alpinelinux/alpine-chroot-install/blob/6d08f12a8a70dd9b9dc7d997c88aa7789cc03c42/alpine-chroot-install#L85-L133
__AlpineKeys='
4a6a0840:MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA1yHJxQgsHQREclQu4Ohe\nqxTxd1tHcNnvnQTu/UrTky8wWvgXT+jpveroeWWnzmsYlDI93eLI2ORakxb3gA2O\nQ0Ry4ws8vhaxLQGC74uQR5+/yYrLuTKydFzuPaS1dK19qJPXB8GMdmFOijnXX4SA\njixuHLe1WW7kZVtjL7nufvpXkWBGjsfrvskdNA/5MfxAeBbqPgaq0QMEfxMAn6/R\nL5kNepi/Vr4S39Xvf2DzWkTLEK8pcnjNkt9/aafhWqFVW7m3HCAII6h/qlQNQKSo\nGuH34Q8GsFG30izUENV9avY7hSLq7nggsvknlNBZtFUcmGoQrtx3FmyYsIC8/R+B\nywIDAQAB
5243ef4b:MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAvNijDxJ8kloskKQpJdx+\nmTMVFFUGDoDCbulnhZMJoKNkSuZOzBoFC94omYPtxnIcBdWBGnrm6ncbKRlR+6oy\nDO0W7c44uHKCFGFqBhDasdI4RCYP+fcIX/lyMh6MLbOxqS22TwSLhCVjTyJeeH7K\naA7vqk+QSsF4TGbYzQDDpg7+6aAcNzg6InNePaywA6hbT0JXbxnDWsB+2/LLSF2G\nmnhJlJrWB1WGjkz23ONIWk85W4S0XB/ewDefd4Ly/zyIciastA7Zqnh7p3Ody6Q0\nsS2MJzo7p3os1smGjUF158s6m/JbVh4DN6YIsxwl2OjDOz9R0OycfJSDaBVIGZzg\ncQIDAQAB
524d27bb:MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAr8s1q88XpuJWLCZALdKj\nlN8wg2ePB2T9aIcaxryYE/Jkmtu+ZQ5zKq6BT3y/udt5jAsMrhHTwroOjIsF9DeG\ne8Y3vjz+Hh4L8a7hZDaw8jy3CPag47L7nsZFwQOIo2Cl1SnzUc6/owoyjRU7ab0p\niWG5HK8IfiybRbZxnEbNAfT4R53hyI6z5FhyXGS2Ld8zCoU/R4E1P0CUuXKEN4p0\n64dyeUoOLXEWHjgKiU1mElIQj3k/IF02W89gDj285YgwqA49deLUM7QOd53QLnx+\nxrIrPv3A+eyXMFgexNwCKQU9ZdmWa00MjjHlegSGK8Y2NPnRoXhzqSP9T9i2HiXL\nVQIDAQAB
5261cecb:MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAwlzMkl7b5PBdfMzGdCT0\ncGloRr5xGgVmsdq5EtJvFkFAiN8Ac9MCFy/vAFmS8/7ZaGOXoCDWbYVLTLOO2qtX\nyHRl+7fJVh2N6qrDDFPmdgCi8NaE+3rITWXGrrQ1spJ0B6HIzTDNEjRKnD4xyg4j\ng01FMcJTU6E+V2JBY45CKN9dWr1JDM/nei/Pf0byBJlMp/mSSfjodykmz4Oe13xB\nCa1WTwgFykKYthoLGYrmo+LKIGpMoeEbY1kuUe04UiDe47l6Oggwnl+8XD1MeRWY\nsWgj8sF4dTcSfCMavK4zHRFFQbGp/YFJ/Ww6U9lA3Vq0wyEI6MCMQnoSMFwrbgZw\nwwIDAQAB
58199dcc:MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA3v8/ye/V/t5xf4JiXLXa\nhWFRozsnmn3hobON20GdmkrzKzO/eUqPOKTpg2GtvBhK30fu5oY5uN2ORiv2Y2ht\neLiZ9HVz3XP8Fm9frha60B7KNu66FO5P2o3i+E+DWTPqqPcCG6t4Znk2BypILcit\nwiPKTsgbBQR2qo/cO01eLLdt6oOzAaF94NH0656kvRewdo6HG4urbO46tCAizvCR\nCA7KGFMyad8WdKkTjxh8YLDLoOCtoZmXmQAiwfRe9pKXRH/XXGop8SYptLqyVVQ+\ntegOD9wRs2tOlgcLx4F/uMzHN7uoho6okBPiifRX+Pf38Vx+ozXh056tjmdZkCaV\naQIDAQAB
58cbb476:MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAoSPnuAGKtRIS5fEgYPXD\n8pSGvKAmIv3A08LBViDUe+YwhilSHbYXUEAcSH1KZvOo1WT1x2FNEPBEFEFU1Eyc\n+qGzbA03UFgBNvArurHQ5Z/GngGqE7IarSQFSoqewYRtFSfp+TL9CUNBvM0rT7vz\n2eMu3/wWG+CBmb92lkmyWwC1WSWFKO3x8w+Br2IFWvAZqHRt8oiG5QtYvcZL6jym\nY8T6sgdDlj+Y+wWaLHs9Fc+7vBuyK9C4O1ORdMPW15qVSl4Lc2Wu1QVwRiKnmA+c\nDsH/m7kDNRHM7TjWnuj+nrBOKAHzYquiu5iB3Qmx+0gwnrSVf27Arc3ozUmmJbLj\nzQIDAQAB
58e4f17d:MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAvBxJN9ErBgdRcPr5g4hV\nqyUSGZEKuvQliq2Z9SRHLh2J43+EdB6A+yzVvLnzcHVpBJ+BZ9RV30EM9guck9sh\nr+bryZcRHyjG2wiIEoduxF2a8KeWeQH7QlpwGhuobo1+gA8L0AGImiA6UP3LOirl\nI0G2+iaKZowME8/tydww4jx5vG132JCOScMjTalRsYZYJcjFbebQQolpqRaGB4iG\nWqhytWQGWuKiB1A22wjmIYf3t96l1Mp+FmM2URPxD1gk/BIBnX7ew+2gWppXOK9j\n1BJpo0/HaX5XoZ/uMqISAAtgHZAqq+g3IUPouxTphgYQRTRYpz2COw3NF43VYQrR\nbQIDAQAB
60ac2099:MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAwR4uJVtJOnOFGchnMW5Y\nj5/waBdG1u5BTMlH+iQMcV5+VgWhmpZHJCBz3ocD+0IGk2I68S5TDOHec/GSC0lv\n6R9o6F7h429GmgPgVKQsc8mPTPtbjJMuLLs4xKc+viCplXc0Nc0ZoHmCH4da6fCV\ntdpHQjVe6F9zjdquZ4RjV6R6JTiN9v924dGMAkbW/xXmamtz51FzondKC52Gh8Mo\n/oA0/T0KsCMCi7tb4QNQUYrf+Xcha9uus4ww1kWNZyfXJB87a2kORLiWMfs2IBBJ\nTmZ2Fnk0JnHDb8Oknxd9PvJPT0mvyT8DA+KIAPqNvOjUXP4bnjEHJcoCP9S5HkGC\nIQIDAQAB
6165ee59:MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAutQkua2CAig4VFSJ7v54\nALyu/J1WB3oni7qwCZD3veURw7HxpNAj9hR+S5N/pNeZgubQvJWyaPuQDm7PTs1+\ntFGiYNfAsiibX6Rv0wci3M+z2XEVAeR9Vzg6v4qoofDyoTbovn2LztaNEjTkB+oK\ntlvpNhg1zhou0jDVYFniEXvzjckxswHVb8cT0OMTKHALyLPrPOJzVtM9C1ew2Nnc\n3848xLiApMu3NBk0JqfcS3Bo5Y2b1FRVBvdt+2gFoKZix1MnZdAEZ8xQzL/a0YS5\nHd0wj5+EEKHfOd3A75uPa/WQmA+o0cBFfrzm69QDcSJSwGpzWrD1ScH3AK8nWvoj\nv7e9gukK/9yl1b4fQQ00vttwJPSgm9EnfPHLAtgXkRloI27H6/PuLoNvSAMQwuCD\nhQRlyGLPBETKkHeodfLoULjhDi1K2gKJTMhtbnUcAA7nEphkMhPWkBpgFdrH+5z4\nLxy+3ek0cqcI7K68EtrffU8jtUj9LFTUC8dERaIBs7NgQ/LfDbDfGh9g6qVj1hZl\nk9aaIPTm/xsi8v3u+0qaq7KzIBc9s59JOoA8TlpOaYdVgSQhHHLBaahOuAigH+VI\nisbC9vmqsThF2QdDtQt37keuqoda2E6sL7PUvIyVXDRfwX7uMDjlzTxHTymvq2Ck\nhtBqojBnThmjJQFgZXocHG8CAwEAAQ==
61666e3f:MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAlEyxkHggKCXC2Wf5Mzx4\nnZLFZvU2bgcA3exfNPO/g1YunKfQY+Jg4fr6tJUUTZ3XZUrhmLNWvpvSwDS19ZmC\nIXOu0+V94aNgnhMsk9rr59I8qcbsQGIBoHzuAl8NzZCgdbEXkiY90w1skUw8J57z\nqCsMBydAueMXuWqF5nGtYbi5vHwK42PffpiZ7G5Kjwn8nYMW5IZdL6ZnMEVJUWC9\nI4waeKg0yskczYDmZUEAtrn3laX9677ToCpiKrvmZYjlGl0BaGp3cxggP2xaDbUq\nqfFxWNgvUAb3pXD09JM6Mt6HSIJaFc9vQbrKB9KT515y763j5CC2KUsilszKi3mB\nHYe5PoebdjS7D1Oh+tRqfegU2IImzSwW3iwA7PJvefFuc/kNIijfS/gH/cAqAK6z\nbhdOtE/zc7TtqW2Wn5Y03jIZdtm12CxSxwgtCF1NPyEWyIxAQUX9ACb3M0FAZ61n\nfpPrvwTaIIxxZ01L3IzPLpbc44x/DhJIEU+iDt6IMTrHOphD9MCG4631eIdB0H1b\n6zbNX1CXTsafqHRFV9XmYYIeOMggmd90s3xIbEujA6HKNP/gwzO6CDJ+nHFDEqoF\nSkxRdTkEqjTjVKieURW7Swv7zpfu5PrsrrkyGnsRrBJJzXlm2FOOxnbI2iSL1B5F\nrO5kbUxFeZUIDq+7Yv4kLWcCAwEAAQ==
616a9724:MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAnC+bR4bHf/L6QdU4puhQ\ngl1MHePszRC38bzvVFDUJsmCaMCL2suCs2A2yxAgGb9pu9AJYLAmxQC4mM3jNqhg\n/E7yuaBbek3O02zN/ctvflJ250wZCy+z0ZGIp1ak6pu1j14IwHokl9j36zNfGtfv\nADVOcdpWITFFlPqwq1qt/H3UsKVmtiF3BNWWTeUEQwKvlU8ymxgS99yn0+4OPyNT\nL3EUeS+NQJtDS01unau0t7LnjUXn+XIneWny8bIYOQCuVR6s/gpIGuhBaUqwaJOw\n7jkJZYF2Ij7uPb4b5/R3vX2FfxxqEHqssFSg8FFUNTZz3qNZs0CRVyfA972g9WkJ\nhPfn31pQYil4QGRibCMIeU27YAEjXoqfJKEPh4UWMQsQLrEfdGfb8VgwrPbniGfU\nL3jKJR3VAafL9330iawzVQDlIlwGl6u77gEXMl9K0pfazunYhAp+BMP+9ot5ckK+\nosmrqj11qMESsAj083GeFdfV3pXEIwUytaB0AKEht9DbqUfiE/oeZ/LAXgySMtVC\nsbC4ESmgVeY2xSBIJdDyUap7FR49GGrw0W49NUv9gRgQtGGaNVQQO9oGL2PBC41P\niWF9GLoX30HIz1P8PF/cZvicSSPkQf2Z6TV+t0ebdGNS5DjapdnCrq8m9Z0pyKsQ\nuxAL2a7zX8l5i1CZh1ycUGsCAwEAAQ==
616abc23:MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEA0MfCDrhODRCIxR9Dep1s\neXafh5CE5BrF4WbCgCsevyPIdvTeyIaW4vmO3bbG4VzhogDZju+R3IQYFuhoXP5v\nY+zYJGnwrgz3r5wYAvPnLEs1+dtDKYOgJXQj+wLJBW1mzRDL8FoRXOe5iRmn1EFS\nwZ1DoUvyu7/J5r0itKicZp3QKED6YoilXed+1vnS4Sk0mzN4smuMR9eO1mMCqNp9\n9KTfRDHTbakIHwasECCXCp50uXdoW6ig/xUAFanpm9LtK6jctNDbXDhQmgvAaLXZ\nLvFqoaYJ/CvWkyYCgL6qxvMvVmPoRv7OPcyni4xR/WgWa0MSaEWjgPx3+yj9fiMA\n1S02pFWFDOr5OUF/O4YhFJvUCOtVsUPPfA/Lj6faL0h5QI9mQhy5Zb9TTaS9jB6p\nLw7u0dJlrjFedk8KTJdFCcaGYHP6kNPnOxMylcB/5WcztXZVQD5WpCicGNBxCGMm\nW64SgrV7M07gQfL/32QLsdqPUf0i8hoVD8wfQ3EpbQzv6Fk1Cn90bZqZafg8XWGY\nwddhkXk7egrr23Djv37V2okjzdqoyLBYBxMz63qQzFoAVv5VoY2NDTbXYUYytOvG\nGJ1afYDRVWrExCech1mX5ZVUB1br6WM+psFLJFoBFl6mDmiYt0vMYBddKISsvwLl\nIJQkzDwtXzT2cSjoj3T5QekCAwEAAQ==
616ac3bc:MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAvaaoSLab+IluixwKV5Od\n0gib2YurjPatGIbn5Ov2DLUFYiebj2oJINXJSwUOO+4WcuHFEqiL/1rya+k5hLZt\nhnPL1tn6QD4rESznvGSasRCQNT2vS/oyZbTYJRyAtFkEYLlq0t3S3xBxxHWuvIf0\nqVxVNYpQWyM3N9RIeYBR/euXKJXileSHk/uq1I5wTC0XBIHWcthczGN0m9wBEiWS\n0m3cnPk4q0Ea8mUJ91Rqob19qETz6VbSPYYpZk3qOycjKosuwcuzoMpwU8KRiMFd\n5LHtX0Hx85ghGsWDVtS0c0+aJa4lOMGvJCAOvDfqvODv7gKlCXUpgumGpLdTmaZ8\n1RwqspAe3IqBcdKTqRD4m2mSg23nVx2FAY3cjFvZQtfooT7q1ItRV5RgH6FhQSl7\n+6YIMJ1Bf8AAlLdRLpg+doOUGcEn+pkDiHFgI8ylH1LKyFKw+eXaAml/7DaWZk1d\ndqggwhXOhc/UUZFQuQQ8A8zpA13PcbC05XxN2hyP93tCEtyynMLVPtrRwDnHxFKa\nqKzs3rMDXPSXRn3ZZTdKH3069ApkEjQdpcwUh+EmJ1Ve/5cdtzT6kKWCjKBFZP/s\n91MlRrX2BTRdHaU5QJkUheUtakwxuHrdah2F94lRmsnQlpPr2YseJu6sIE+Dnx4M\nCfhdVbQL2w54R645nlnohu8CAwEAAQ==
616adfeb:MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAq0BFD1D4lIxQcsqEpQzU\npNCYM3aP1V/fxxVdT4DWvSI53JHTwHQamKdMWtEXetWVbP5zSROniYKFXd/xrD9X\n0jiGHey3lEtylXRIPxe5s+wXoCmNLcJVnvTcDtwx/ne2NLHxp76lyc25At+6RgE6\nADjLVuoD7M4IFDkAsd8UQ8zM0Dww9SylIk/wgV3ZkifecvgUQRagrNUdUjR56EBZ\nraQrev4hhzOgwelT0kXCu3snbUuNY/lU53CoTzfBJ5UfEJ5pMw1ij6X0r5S9IVsy\nKLWH1hiO0NzU2c8ViUYCly4Fe9xMTFc6u2dy/dxf6FwERfGzETQxqZvSfrRX+GLj\n/QZAXiPg5178hT/m0Y3z5IGenIC/80Z9NCi+byF1WuJlzKjDcF/TU72zk0+PNM/H\nKuppf3JT4DyjiVzNC5YoWJT2QRMS9KLP5iKCSThwVceEEg5HfhQBRT9M6KIcFLSs\nmFjx9kNEEmc1E8hl5IR3+3Ry8G5/bTIIruz14jgeY9u5jhL8Vyyvo41jgt9sLHR1\n/J1TxKfkgksYev7PoX6/ZzJ1ksWKZY5NFoDXTNYUgzFUTOoEaOg3BAQKadb3Qbbq\nXIrxmPBdgrn9QI7NCgfnAY3Tb4EEjs3ON/BNyEhUENcXOH6I1NbcuBQ7g9P73kE4\nVORdoc8MdJ5eoKBpO8Ww8HECAwEAAQ==
616ae350:MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAyduVzi1mWm+lYo2Tqt/0\nXkCIWrDNP1QBMVPrE0/ZlU2bCGSoo2Z9FHQKz/mTyMRlhNqTfhJ5qU3U9XlyGOPJ\npiM+b91g26pnpXJ2Q2kOypSgOMOPA4cQ42PkHBEqhuzssfj9t7x47ppS94bboh46\nxLSDRff/NAbtwTpvhStV3URYkxFG++cKGGa5MPXBrxIp+iZf9GnuxVdST5PGiVGP\nODL/b69sPJQNbJHVquqUTOh5Ry8uuD2WZuXfKf7/C0jC/ie9m2+0CttNu9tMciGM\nEyKG1/Xhk5iIWO43m4SrrT2WkFlcZ1z2JSf9Pjm4C2+HovYpihwwdM/OdP8Xmsnr\nDzVB4YvQiW+IHBjStHVuyiZWc+JsgEPJzisNY0Wyc/kNyNtqVKpX6dRhMLanLmy+\nf53cCSI05KPQAcGj6tdL+D60uKDkt+FsDa0BTAobZ31OsFVid0vCXtsbplNhW1IF\nHwsGXBTVcfXg44RLyL8Lk/2dQxDHNHzAUslJXzPxaHBLmt++2COa2EI1iWlvtznk\nOk9WP8SOAIj+xdqoiHcC4j72BOVVgiITIJNHrbppZCq6qPR+fgXmXa+sDcGh30m6\n9Wpbr28kLMSHiENCWTdsFij+NQTd5S47H7XTROHnalYDuF1RpS+DpQidT5tUimaT\nJZDr++FjKrnnijbyNF8b98UCAwEAAQ==
616db30d:MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAnpUpyWDWjlUk3smlWeA0\nlIMW+oJ38t92CRLHH3IqRhyECBRW0d0aRGtq7TY8PmxjjvBZrxTNDpJT6KUk4LRm\na6A6IuAI7QnNK8SJqM0DLzlpygd7GJf8ZL9SoHSH+gFsYF67Cpooz/YDqWrlN7Vw\ntO00s0B+eXy+PCXYU7VSfuWFGK8TGEv6HfGMALLjhqMManyvfp8hz3ubN1rK3c8C\nUS/ilRh1qckdbtPvoDPhSbTDmfU1g/EfRSIEXBrIMLg9ka/XB9PvWRrekrppnQzP\nhP9YE3x/wbFc5QqQWiRCYyQl/rgIMOXvIxhkfe8H5n1Et4VAorkpEAXdsfN8KSVv\nLSMazVlLp9GYq5SUpqYX3KnxdWBgN7BJoZ4sltsTpHQ/34SXWfu3UmyUveWj7wp0\nx9hwsPirVI00EEea9AbP7NM2rAyu6ukcm4m6ATd2DZJIViq2es6m60AE6SMCmrQF\nwmk4H/kdQgeAELVfGOm2VyJ3z69fQuywz7xu27S6zTKi05Qlnohxol4wVb6OB7qG\nLPRtK9ObgzRo/OPumyXqlzAi/Yvyd1ZQk8labZps3e16bQp8+pVPiumWioMFJDWV\nGZjCmyMSU8V6MB6njbgLHoyg2LCukCAeSjbPGGGYhnKLm1AKSoJh3IpZuqcKCk5C\n8CM1S15HxV78s9dFntEqIokCAwEAAQ==
66ba20fe:MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAtfB12w4ZgqsXWZDfUAV/\n6Y4aHUKIu3q4SXrNZ7CXF9nXoAVYrS7NAxJdAodsY3vPCN0g5O8DFXR+390LdOuQ\n+HsGKCc1k5tX5ZXld37EZNTNSbR0k+NKhd9h6X3u6wqPOx7SIKxwAQR8qeeFq4pP\nrt9GAGlxtuYgzIIcKJPwE0dZlcBCg+GnptCUZXp/38BP1eYC+xTXSL6Muq1etYfg\nodXdb7Yl+2h1IHuOwo5rjgY5kpY7GcAs8AjGk3lDD/av60OTYccknH0NCVSmPoXK\nvrxDBOn0LQRNBLcAfnTKgHrzy0Q5h4TNkkyTgxkoQw5ObDk9nnabTxql732yy9BY\ns+hM9+dSFO1HKeVXreYSA2n1ndF18YAvAumzgyqzB7I4pMHXq1kC/8bONMJxwSkS\nYm6CoXKyavp7RqGMyeVpRC7tV+blkrrUml0BwNkxE+XnwDRB3xDV6hqgWe0XrifD\nYTfvd9ScZQP83ip0r4IKlq4GMv/R5shcCRJSkSZ6QSGshH40JYSoiwJf5FHbj9ND\n7do0UAqebWo4yNx63j/wb2ULorW3AClv0BCFSdPsIrCStiGdpgJDBR2P2NZOCob3\nG9uMj+wJD6JJg2nWqNJxkANXX37Qf8plgzssrhrgOvB0fjjS7GYhfkfmZTJ0wPOw\nA8+KzFseBh4UFGgue78KwgkCAwEAAQ==
'
__Keyring=
__KeyringFile="/usr/share/keyrings/ubuntu-archive-keyring.gpg"
__SkipSigCheck=0
__SkipEmulation=0
__UseMirror=0

__UnprocessedBuildArgs=
while :; do
    if [[ "$#" -le 0 ]]; then
        break
    fi

    lowerI="$(echo "$1" | tr "[:upper:]" "[:lower:]")"
    case $lowerI in
        -\?|-h|--help)
            usage
            ;;
        arm)
            __BuildArch=arm
            __UbuntuArch=armhf
            __AlpineArch=armv7
            __QEMUArch=arm
            ;;
        arm64)
            __BuildArch=arm64
            __UbuntuArch=arm64
            __AlpineArch=aarch64
            __QEMUArch=aarch64
            __FreeBSDArch=arm64
            __FreeBSDMachineArch=aarch64
            ;;
        armel)
            __BuildArch=armel
            __UbuntuArch=armel
            __UbuntuRepo="http://ftp.debian.org/debian/"
            __CodeName=jessie
            __KeyringFile="/usr/share/keyrings/debian-archive-keyring.gpg"
            ;;
        armv6)
            __BuildArch=armv6
            __UbuntuArch=armhf
            __QEMUArch=arm
            __UbuntuRepo="http://raspbian.raspberrypi.org/raspbian/"
            __CodeName=buster
            __KeyringFile="/usr/share/keyrings/raspbian-archive-keyring.gpg"
            __LLDB_Package="liblldb-6.0-dev"
            __UbuntuSuites=

            if [[ -e "$__KeyringFile" ]]; then
                __Keyring="--keyring $__KeyringFile"
            fi
            ;;
        loongarch64)
            __BuildArch=loongarch64
            __AlpineArch=loongarch64
            __QEMUArch=loongarch64
            __UbuntuArch=loong64
            __UbuntuSuites=
            __DebianSuites=unreleased
            __LLDB_Package="liblldb-19-dev"

            if [[ "$__CodeName" == "sid" ]]; then
                __UbuntuRepo="http://ftp.ports.debian.org/debian-ports/"
            fi
            ;;
        riscv64)
            __BuildArch=riscv64
            __AlpineArch=riscv64
            __AlpinePackages="${__AlpinePackages// lldb-dev/}"
            __QEMUArch=riscv64
            __UbuntuArch=riscv64
            __UbuntuPackages="${__UbuntuPackages// libunwind8-dev/}"
            unset __LLDB_Package
            ;;
        ppc64le)
            __BuildArch=ppc64le
            __AlpineArch=ppc64le
            __QEMUArch=ppc64le
            __UbuntuArch=ppc64el
            __UbuntuRepo="http://ports.ubuntu.com/ubuntu-ports/"
            __UbuntuPackages="${__UbuntuPackages// libunwind8-dev/}"
            __UbuntuPackages="${__UbuntuPackages// libomp-dev/}"
            __UbuntuPackages="${__UbuntuPackages// libomp5/}"
            unset __LLDB_Package
            ;;
        s390x)
            __BuildArch=s390x
            __AlpineArch=s390x
            __QEMUArch=s390x
            __UbuntuArch=s390x
            __UbuntuRepo="http://ports.ubuntu.com/ubuntu-ports/"
            __UbuntuPackages="${__UbuntuPackages// libunwind8-dev/}"
            __UbuntuPackages="${__UbuntuPackages// libomp-dev/}"
            __UbuntuPackages="${__UbuntuPackages// libomp5/}"
            unset __LLDB_Package
            ;;
        x64)
            __BuildArch=x64
            __AlpineArch=x86_64
            __UbuntuArch=amd64
            __FreeBSDArch=amd64
            __FreeBSDMachineArch=amd64
            __illumosArch=x86_64
            __HaikuArch=x86_64
            __UbuntuRepo="http://archive.ubuntu.com/ubuntu/"
            ;;
        x86)
            __BuildArch=x86
            __UbuntuArch=i386
            __AlpineArch=x86
            __UbuntuRepo="http://archive.ubuntu.com/ubuntu/"
            ;;
        lldb*)
            version="$(echo "$lowerI" | tr -d '[:alpha:]-=')"
            majorVersion="${version%%.*}"

            [ -z "${version##*.*}" ] && minorVersion="${version#*.}"
            if [ -z "$minorVersion" ]; then
                minorVersion=0
            fi

            # for versions > 6.0, lldb has dropped the minor version
            if [ "$majorVersion" -le 6 ]; then
                version="$majorVersion.$minorVersion"
            else
                version="$majorVersion"
            fi

            __LLDB_Package="liblldb-${version}-dev"
            ;;
        no-lldb)
            unset __LLDB_Package
            ;;
        llvm*)
            version="$(echo "$lowerI" | tr -d '[:alpha:]-=')"
            __LLVM_MajorVersion="${version%%.*}"

            [ -z "${version##*.*}" ] && __LLVM_MinorVersion="${version#*.}"
            if [ -z "$__LLVM_MinorVersion" ]; then
                __LLVM_MinorVersion=0
            fi

            # for versions > 6.0, lldb has dropped the minor version
            if [ "$__LLVM_MajorVersion" -gt 6 ]; then
                __LLVM_MinorVersion=
            fi

            ;;
        xenial) # Ubuntu 16.04
            if [[ "$__CodeName" != "jessie" ]]; then
                __CodeName=xenial
            fi
            ;;
        zesty) # Ubuntu 17.04
            if [[ "$__CodeName" != "jessie" ]]; then
                __CodeName=zesty
            fi
            ;;
        bionic) # Ubuntu 18.04
            if [[ "$__CodeName" != "jessie" ]]; then
                __CodeName=bionic
            fi
            ;;
        focal) # Ubuntu 20.04
            if [[ "$__CodeName" != "jessie" ]]; then
                __CodeName=focal
            fi
            ;;
        jammy) # Ubuntu 22.04
            if [[ "$__CodeName" != "jessie" ]]; then
                __CodeName=jammy
            fi
            ;;
        noble) # Ubuntu 24.04
            if [[ "$__CodeName" != "jessie" ]]; then
                __CodeName=noble
            fi
            if [[ -n "$__LLDB_Package" ]]; then
                __LLDB_Package="liblldb-18-dev"
            fi
            ;;
        jessie) # Debian 8
            __CodeName=jessie
            __KeyringFile="/usr/share/keyrings/debian-archive-keyring.gpg"

            if [[ -z "$__UbuntuRepo" ]]; then
                __UbuntuRepo="http://ftp.debian.org/debian/"
            fi
            ;;
        stretch) # Debian 9
            __CodeName=stretch
            __LLDB_Package="liblldb-6.0-dev"
            __KeyringFile="/usr/share/keyrings/debian-archive-keyring.gpg"

            if [[ -z "$__UbuntuRepo" ]]; then
                __UbuntuRepo="http://ftp.debian.org/debian/"
            fi
            ;;
        buster) # Debian 10
            __CodeName=buster
            __LLDB_Package="liblldb-6.0-dev"
            __KeyringFile="/usr/share/keyrings/debian-archive-keyring.gpg"

            if [[ -z "$__UbuntuRepo" ]]; then
                __UbuntuRepo="http://ftp.debian.org/debian/"
            fi
            ;;
        bullseye) # Debian 11
            __CodeName=bullseye
            __KeyringFile="/usr/share/keyrings/debian-archive-keyring.gpg"

            if [[ -z "$__UbuntuRepo" ]]; then
                __UbuntuRepo="http://ftp.debian.org/debian/"
            fi
            ;;
        bookworm) # Debian 12
            __CodeName=bookworm
            __KeyringFile="/usr/share/keyrings/debian-archive-keyring.gpg"

            if [[ -z "$__UbuntuRepo" ]]; then
                __UbuntuRepo="http://ftp.debian.org/debian/"
            fi
            ;;
        sid) # Debian sid
            __CodeName=sid
            __UbuntuSuites=

            # Debian-Ports architectures need different values
            case "$__UbuntuArch" in
            amd64|arm64|armel|armhf|i386|mips64el|ppc64el|riscv64|s390x)
                __KeyringFile="/usr/share/keyrings/debian-archive-keyring.gpg"

                if [[ -z "$__UbuntuRepo" ]]; then
                    __UbuntuRepo="http://ftp.debian.org/debian/"
                fi
                ;;
            *)
                __KeyringFile="/usr/share/keyrings/debian-ports-archive-keyring.gpg"

                if [[ -z "$__UbuntuRepo" ]]; then
                    __UbuntuRepo="http://ftp.ports.debian.org/debian-ports/"
                fi
                ;;
            esac

            if [[ -e "$__KeyringFile" ]]; then
                __Keyring="--keyring $__KeyringFile"
            fi
            ;;
        tizen)
            __CodeName=
            __UbuntuRepo=
            __Tizen=tizen
            ;;
        alpine*)
            __CodeName=alpine
            __UbuntuRepo=

            if [[ "$lowerI" == "alpineedge" ]]; then
                __AlpineVersion=edge
            else
                version="$(echo "$lowerI" | tr -d '[:alpha:]-=')"
                __AlpineMajorVersion="${version%%.*}"
                __AlpineMinorVersion="${version#*.}"
                __AlpineVersion="$__AlpineMajorVersion.$__AlpineMinorVersion"
            fi
            ;;
        freebsd13)
            __CodeName=freebsd
            __SkipUnmount=1
            ;;
        freebsd14)
            __CodeName=freebsd
            __FreeBSDBase="14.2-RELEASE"
            __FreeBSDABI="14"
            __SkipUnmount=1
            ;;
        illumos)
            __CodeName=illumos
            __SkipUnmount=1
            ;;
        haiku)
            __CodeName=haiku
            __SkipUnmount=1
            ;;
        --skipunmount)
            __SkipUnmount=1
            ;;
        --skipsigcheck)
            __SkipSigCheck=1
            ;;
        --skipemulation)
            __SkipEmulation=1
            ;;
        --rootfsdir|-rootfsdir)
            shift
            __RootfsDir="$1"
            ;;
        --use-mirror)
            __UseMirror=1
            ;;
        --use-jobs)
            shift
            MAXJOBS=$1
            ;;
        *)
            __UnprocessedBuildArgs="$__UnprocessedBuildArgs $1"
            ;;
    esac

    shift
done

case "$__AlpineVersion" in
    3.14) __AlpinePackages+=" llvm11-libs" ;;
    3.15) __AlpinePackages+=" llvm12-libs" ;;
    3.16) __AlpinePackages+=" llvm13-libs" ;;
    3.17) __AlpinePackages+=" llvm15-libs" ;;
    edge) __AlpineLlvmLibsLookup=1 ;;
    *)
        if [[ "$__AlpineArch" =~ s390x|ppc64le ]]; then
            __AlpineVersion=3.15 # minimum version that supports lldb-dev
            __AlpinePackages+=" llvm12-libs"
        elif [[ "$__AlpineArch" == "x86" ]]; then
            __AlpineVersion=3.17 # minimum version that supports lldb-dev
            __AlpinePackages+=" llvm15-libs"
        elif [[ "$__AlpineArch" == "riscv64" || "$__AlpineArch" == "loongarch64" ]]; then
            __AlpineVersion=3.21 # minimum version that supports lldb-dev
            __AlpinePackages+=" llvm19-libs"
        elif [[ -n "$__AlpineMajorVersion" ]]; then
            # use whichever alpine version is provided and select the latest toolchain libs
            __AlpineLlvmLibsLookup=1
        else
            __AlpineVersion=3.13 # 3.13 to maximize compatibility
            __AlpinePackages+=" llvm10-libs"
        fi
esac

if [[ "$__AlpineVersion" =~ 3\.1[345] ]]; then
    # compiler-rt--static was merged in compiler-rt package in alpine 3.16
    # for older versions, we need compiler-rt--static, so replace the name
    __AlpinePackages="${__AlpinePackages/compiler-rt/compiler-rt-static}"
fi

if [[ "$__BuildArch" == "armel" ]]; then
    __LLDB_Package="lldb-3.5-dev"
fi

__UbuntuPackages+=" ${__LLDB_Package:-}"

if [[ -z "$__UbuntuRepo" ]]; then
    __UbuntuRepo="http://ports.ubuntu.com/"
fi

if [[ -n "$__LLVM_MajorVersion" ]]; then
    __UbuntuPackages+=" libclang-common-${__LLVM_MajorVersion}${__LLVM_MinorVersion:+.$__LLVM_MinorVersion}-dev"
fi

if [[ -z "$__RootfsDir" && -n "$ROOTFS_DIR" ]]; then
    __RootfsDir="$ROOTFS_DIR"
fi

if [[ -z "$__RootfsDir" ]]; then
    __RootfsDir="$__CrossDir/../../../.tools/rootfs/$__BuildArch"
fi

if [[ -d "$__RootfsDir" ]]; then
    if [[ "$__SkipUnmount" == "0" ]]; then
        umount "$__RootfsDir"/* || true
    fi
    rm -rf "$__RootfsDir"
fi

mkdir -p "$__RootfsDir"
__RootfsDir="$( cd "$__RootfsDir" && pwd )"

__hasWget=
ensureDownloadTool()
{
    if command -v wget &> /dev/null; then
        __hasWget=1
    elif command -v curl &> /dev/null; then
        __hasWget=0
    else
        >&2 echo "ERROR: either wget or curl is required by this script."
        exit 1
    fi
}

if [[ "$__CodeName" == "alpine" ]]; then
    __ApkToolsVersion=2.12.11
    __ApkToolsDir="$(mktemp -d)"
    __ApkKeysDir="$(mktemp -d)"
    arch="$(uname -m)"

    ensureDownloadTool

    if [[ "$__hasWget" == 1 ]]; then
        wget -P "$__ApkToolsDir" "https://gitlab.alpinelinux.org/api/v4/projects/5/packages/generic/v$__ApkToolsVersion/$arch/apk.static"
    else
        curl -SLO --create-dirs --output-dir "$__ApkToolsDir" "https://gitlab.alpinelinux.org/api/v4/projects/5/packages/generic/v$__ApkToolsVersion/$arch/apk.static"
    fi
    if [[ "$arch" == "x86_64" ]]; then
      __ApkToolsSHA512SUM="53e57b49230da07ef44ee0765b9592580308c407a8d4da7125550957bb72cb59638e04f8892a18b584451c8d841d1c7cb0f0ab680cc323a3015776affaa3be33"
    elif [[ "$arch" == "aarch64" ]]; then
      __ApkToolsSHA512SUM="9e2b37ecb2b56c05dad23d379be84fd494c14bd730b620d0d576bda760588e1f2f59a7fcb2f2080577e0085f23a0ca8eadd993b4e61c2ab29549fdb71969afd0"
    else
      echo "WARNING: add missing hash for your host architecture. To find the value, use: 'find /tmp -name apk.static -exec sha512sum {} \;'"
    fi
    echo "$__ApkToolsSHA512SUM $__ApkToolsDir/apk.static" | sha512sum -c
    chmod +x "$__ApkToolsDir/apk.static"

    if [[ "$__AlpineVersion" == "edge" ]]; then
        version=edge
    else
        version="v$__AlpineVersion"
    fi

    for line in $__AlpineKeys; do
        id="${line%%:*}"
        content="${line#*:}"

        echo -e "-----BEGIN PUBLIC KEY-----\n$content\n-----END PUBLIC KEY-----" > "$__ApkKeysDir/alpine-devel@lists.alpinelinux.org-$id.rsa.pub"
    done

    if [[ "$__SkipSigCheck" == "1" ]]; then
        __ApkSignatureArg="--allow-untrusted"
    else
        __ApkSignatureArg="--keys-dir $__ApkKeysDir"
    fi

    if [[ "$__SkipEmulation" == "1" ]]; then
        __NoEmulationArg="--no-scripts"
    fi

    # initialize DB
    # shellcheck disable=SC2086
    "$__ApkToolsDir/apk.static" \
        -X "http://dl-cdn.alpinelinux.org/alpine/$version/main" \
        -X "http://dl-cdn.alpinelinux.org/alpine/$version/community" \
        -U $__ApkSignatureArg --root "$__RootfsDir" --arch "$__AlpineArch" --initdb add

    if [[ "$__AlpineLlvmLibsLookup" == 1 ]]; then
        # shellcheck disable=SC2086
        __AlpinePackages+=" $("$__ApkToolsDir/apk.static" \
            -X "http://dl-cdn.alpinelinux.org/alpine/$version/main" \
            -X "http://dl-cdn.alpinelinux.org/alpine/$version/community" \
            -U $__ApkSignatureArg --root "$__RootfsDir" --arch "$__AlpineArch" \
            search 'llvm*-libs' | grep -E '^llvm' | sort | tail -1 | sed 's/-[^-]*//2g')"
    fi

    # install all packages in one go
    # shellcheck disable=SC2086
    "$__ApkToolsDir/apk.static" \
        -X "http://dl-cdn.alpinelinux.org/alpine/$version/main" \
        -X "http://dl-cdn.alpinelinux.org/alpine/$version/community" \
        -U $__ApkSignatureArg --root "$__RootfsDir" --arch "$__AlpineArch" $__NoEmulationArg \
        add $__AlpinePackages

    rm -r "$__ApkToolsDir"
elif [[ "$__CodeName" == "freebsd" ]]; then
    mkdir -p "$__RootfsDir"/usr/local/etc
    JOBS=${MAXJOBS:="$(getconf _NPROCESSORS_ONLN)"}

    ensureDownloadTool

    if [[ "$__hasWget" == 1 ]]; then
        wget -O- "https://download.freebsd.org/ftp/releases/${__FreeBSDArch}/${__FreeBSDMachineArch}/${__FreeBSDBase}/base.txz" | tar -C "$__RootfsDir" -Jxf - ./lib ./usr/lib ./usr/libdata ./usr/include ./usr/share/keys ./etc ./bin/freebsd-version
    else
        curl -SL "https://download.freebsd.org/ftp/releases/${__FreeBSDArch}/${__FreeBSDMachineArch}/${__FreeBSDBase}/base.txz" | tar -C "$__RootfsDir" -Jxf - ./lib ./usr/lib ./usr/libdata ./usr/include ./usr/share/keys ./etc ./bin/freebsd-version
    fi
    echo "ABI = \"FreeBSD:${__FreeBSDABI}:${__FreeBSDMachineArch}\"; FINGERPRINTS = \"${__RootfsDir}/usr/share/keys\"; REPOS_DIR = [\"${__RootfsDir}/etc/pkg\"]; REPO_AUTOUPDATE = NO; RUN_SCRIPTS = NO;" > "${__RootfsDir}"/usr/local/etc/pkg.conf
    echo "FreeBSD: { url: \"pkg+http://pkg.FreeBSD.org/\${ABI}/quarterly\", mirror_type: \"srv\", signature_type: \"fingerprints\", fingerprints: \"/usr/share/keys/pkg\", enabled: yes }" > "${__RootfsDir}"/etc/pkg/FreeBSD.conf
    mkdir -p "$__RootfsDir"/tmp
    # get and build package manager
    if [[ "$__hasWget" == 1 ]]; then
        wget -O- "https://github.com/freebsd/pkg/archive/${__FreeBSDPkg}.tar.gz" | tar -C "$__RootfsDir"/tmp -zxf -
    else
        curl -SL "https://github.com/freebsd/pkg/archive/${__FreeBSDPkg}.tar.gz" | tar -C "$__RootfsDir"/tmp -zxf -
    fi
    cd "$__RootfsDir/tmp/pkg-${__FreeBSDPkg}"
    # needed for install to succeed
    mkdir -p "$__RootfsDir"/host/etc
    ./autogen.sh && ./configure --prefix="$__RootfsDir"/host && make -j "$JOBS" && make install
    rm -rf "$__RootfsDir/tmp/pkg-${__FreeBSDPkg}"
    # install packages we need.
    INSTALL_AS_USER=$(whoami) "$__RootfsDir"/host/sbin/pkg -r "$__RootfsDir" -C "$__RootfsDir"/usr/local/etc/pkg.conf update
    # shellcheck disable=SC2086
    INSTALL_AS_USER=$(whoami) "$__RootfsDir"/host/sbin/pkg -r "$__RootfsDir" -C "$__RootfsDir"/usr/local/etc/pkg.conf install --yes $__FreeBSDPackages
elif [[ "$__CodeName" == "illumos" ]]; then
    mkdir "$__RootfsDir/tmp"
    pushd "$__RootfsDir/tmp"
    JOBS=${MAXJOBS:="$(getconf _NPROCESSORS_ONLN)"}

    ensureDownloadTool

    echo "Downloading sysroot."
    if [[ "$__hasWget" == 1 ]]; then
        wget -O- https://github.com/illumos/sysroot/releases/download/20181213-de6af22ae73b-v1/illumos-sysroot-i386-20181213-de6af22ae73b-v1.tar.gz | tar -C "$__RootfsDir" -xzf -
    else
        curl -SL https://github.com/illumos/sysroot/releases/download/20181213-de6af22ae73b-v1/illumos-sysroot-i386-20181213-de6af22ae73b-v1.tar.gz | tar -C "$__RootfsDir" -xzf -
    fi
    echo "Building binutils. Please wait.."
    if [[ "$__hasWget" == 1 ]]; then
        wget -O- https://ftp.gnu.org/gnu/binutils/binutils-2.42.tar.xz | tar -xJf -
    else
        curl -SL https://ftp.gnu.org/gnu/binutils/binutils-2.42.tar.xz | tar -xJf -
    fi
    mkdir build-binutils && cd build-binutils
    ../binutils-2.42/configure --prefix="$__RootfsDir" --target="${__illumosArch}-sun-solaris2.11" --program-prefix="${__illumosArch}-illumos-" --with-sysroot="$__RootfsDir"
    make -j "$JOBS" && make install && cd ..
    echo "Building gcc. Please wait.."
    if [[ "$__hasWget" == 1 ]]; then
        wget -O- https://ftp.gnu.org/gnu/gcc/gcc-13.3.0/gcc-13.3.0.tar.xz | tar -xJf -
    else
        curl -SL https://ftp.gnu.org/gnu/gcc/gcc-13.3.0/gcc-13.3.0.tar.xz | tar -xJf -
    fi
    CFLAGS="-fPIC"
    CXXFLAGS="-fPIC"
    CXXFLAGS_FOR_TARGET="-fPIC"
    CFLAGS_FOR_TARGET="-fPIC"
    export CFLAGS CXXFLAGS CXXFLAGS_FOR_TARGET CFLAGS_FOR_TARGET
    mkdir build-gcc && cd build-gcc
    ../gcc-13.3.0/configure --prefix="$__RootfsDir" --target="${__illumosArch}-sun-solaris2.11" --program-prefix="${__illumosArch}-illumos-" --with-sysroot="$__RootfsDir" --with-gnu-as       \
        --with-gnu-ld --disable-nls --disable-libgomp --disable-libquadmath --disable-libssp --disable-libvtv --disable-libcilkrts --disable-libada --disable-libsanitizer \
        --disable-libquadmath-support --disable-shared --enable-tls
    make -j "$JOBS" && make install && cd ..
    BaseUrl=https://pkgsrc.smartos.org
    if [[ "$__UseMirror" == 1 ]]; then
        BaseUrl=https://pkgsrc.smartos.skylime.net
    fi
    BaseUrl="$BaseUrl/packages/SmartOS/2019Q4/${__illumosArch}/All"
    echo "Downloading manifest"
    if [[ "$__hasWget" == 1 ]]; then
        wget "$BaseUrl"
    else
        curl -SLO "$BaseUrl"
    fi
    echo "Downloading dependencies."
    read -ra array <<<"$__IllumosPackages"
    for package in "${array[@]}"; do
        echo "Installing '$package'"
        # find last occurrence of package in listing and extract its name
        package="$(sed -En '/.*href="('"$package"'-[0-9].*).tgz".*/h;$!d;g;s//\1/p' All)"
        echo "Resolved name '$package'"
        if [[ "$__hasWget" == 1 ]]; then
            wget "$BaseUrl"/"$package".tgz
        else
            curl -SLO "$BaseUrl"/"$package".tgz
        fi
        ar -x "$package".tgz
        tar --skip-old-files -xzf "$package".tmp.tg* -C "$__RootfsDir" 2>/dev/null
    done
    echo "Cleaning up temporary files."
    popd
    rm -rf "$__RootfsDir"/{tmp,+*}
    mkdir -p "$__RootfsDir"/usr/include/net
    mkdir -p "$__RootfsDir"/usr/include/netpacket
    if [[ "$__hasWget" == 1 ]]; then
        wget -P "$__RootfsDir"/usr/include/net https://raw.githubusercontent.com/illumos/illumos-gate/master/usr/src/uts/common/io/bpf/net/bpf.h
        wget -P "$__RootfsDir"/usr/include/net https://raw.githubusercontent.com/illumos/illumos-gate/master/usr/src/uts/common/io/bpf/net/dlt.h
        wget -P "$__RootfsDir"/usr/include/netpacket https://raw.githubusercontent.com/illumos/illumos-gate/master/usr/src/uts/common/inet/sockmods/netpacket/packet.h
        wget -P "$__RootfsDir"/usr/include/sys https://raw.githubusercontent.com/illumos/illumos-gate/master/usr/src/uts/common/sys/sdt.h
    else
        curl -SLO --create-dirs --output-dir "$__RootfsDir"/usr/include/net https://raw.githubusercontent.com/illumos/illumos-gate/master/usr/src/uts/common/io/bpf/net/bpf.h
        curl -SLO --create-dirs --output-dir "$__RootfsDir"/usr/include/net https://raw.githubusercontent.com/illumos/illumos-gate/master/usr/src/uts/common/io/bpf/net/dlt.h
        curl -SLO --create-dirs --output-dir "$__RootfsDir"/usr/include/netpacket https://raw.githubusercontent.com/illumos/illumos-gate/master/usr/src/uts/common/inet/sockmods/netpacket/packet.h
        curl -SLO --create-dirs --output-dir "$__RootfsDir"/usr/include/sys https://raw.githubusercontent.com/illumos/illumos-gate/master/usr/src/uts/common/sys/sdt.h
    fi
elif [[ "$__CodeName" == "haiku" ]]; then
    JOBS=${MAXJOBS:="$(getconf _NPROCESSORS_ONLN)"}

    echo "Building Haiku sysroot for $__HaikuArch"
    mkdir -p "$__RootfsDir/tmp"
    pushd "$__RootfsDir/tmp"

    mkdir "$__RootfsDir/tmp/download"

    ensureDownloadTool

    echo "Downloading Haiku package tools"
    git clone https://github.com/haiku/haiku-toolchains-ubuntu --depth 1 "$__RootfsDir/tmp/script"
    if [[ "$__hasWget" == 1 ]]; then
        wget -O "$__RootfsDir/tmp/download/hosttools.zip" "$("$__RootfsDir/tmp/script/fetch.sh" --hosttools)"
    else
        curl -SLo "$__RootfsDir/tmp/download/hosttools.zip" "$("$__RootfsDir/tmp/script/fetch.sh" --hosttools)"
    fi

    unzip -o "$__RootfsDir/tmp/download/hosttools.zip" -d "$__RootfsDir/tmp/bin"

    HaikuBaseUrl="https://eu.hpkg.haiku-os.org/haiku/master/$__HaikuArch/current"
    HaikuPortsBaseUrl="https://eu.hpkg.haiku-os.org/haikuports/master/$__HaikuArch/current"

    echo "Downloading HaikuPorts package repository index..."
    if [[ "$__hasWget" == 1 ]]; then
        wget -P "$__RootfsDir/tmp/download" "$HaikuPortsBaseUrl/repo"
    else
        curl -SLO --create-dirs --output-dir "$__RootfsDir/tmp/download" "$HaikuPortsBaseUrl/repo"
    fi

    echo "Downloading Haiku packages"
    read -ra array <<<"$__HaikuPackages"
    for package in "${array[@]}"; do
        echo "Downloading $package..."
        hpkgFilename="$(LD_LIBRARY_PATH="$__RootfsDir/tmp/bin" "$__RootfsDir/tmp/bin/package_repo" list -f "$__RootfsDir/tmp/download/repo" |
            grep -E "${package}-" | sort -V | tail -n 1 | xargs)"
        if [ -z "$hpkgFilename" ]; then
            >&2 echo "ERROR: package $package missing."
            exit 1
        fi
        echo "Resolved filename: $hpkgFilename..."
        hpkgDownloadUrl="$HaikuPortsBaseUrl/packages/$hpkgFilename"
        if [[ "$__hasWget" == 1 ]]; then
            wget -P "$__RootfsDir/tmp/download" "$hpkgDownloadUrl"
        else
            curl -SLO --create-dirs --output-dir "$__RootfsDir/tmp/download" "$hpkgDownloadUrl"
        fi
    done
    for package in haiku haiku_devel; do
        echo "Downloading $package..."
        if [[ "$__hasWget" == 1 ]]; then
            hpkgVersion="$(wget -qO- "$HaikuBaseUrl" | sed -n 's/^.*version: "\([^"]*\)".*$/\1/p')"
            wget -P "$__RootfsDir/tmp/download" "$HaikuBaseUrl/packages/$package-$hpkgVersion-1-$__HaikuArch.hpkg"
        else
            hpkgVersion="$(curl -sSL "$HaikuBaseUrl" | sed -n 's/^.*version: "\([^"]*\)".*$/\1/p')"
            curl -SLO --create-dirs --output-dir "$__RootfsDir/tmp/download" "$HaikuBaseUrl/packages/$package-$hpkgVersion-1-$__HaikuArch.hpkg"
        fi
    done

    # Set up the sysroot
    echo "Setting up sysroot and extracting required packages"
    mkdir -p "$__RootfsDir/boot/system"
    for file in "$__RootfsDir/tmp/download/"*.hpkg; do
        echo "Extracting $file..."
        LD_LIBRARY_PATH="$__RootfsDir/tmp/bin" "$__RootfsDir/tmp/bin/package" extract -C "$__RootfsDir/boot/system" "$file"
    done

    # Download buildtools
    echo "Downloading Haiku buildtools"
    if [[ "$__hasWget" == 1 ]]; then
        wget -O "$__RootfsDir/tmp/download/buildtools.zip" "$("$__RootfsDir/tmp/script/fetch.sh" --buildtools --arch=$__HaikuArch)"
    else
        curl -SLo "$__RootfsDir/tmp/download/buildtools.zip" "$("$__RootfsDir/tmp/script/fetch.sh" --buildtools --arch=$__HaikuArch)"
    fi
    unzip -o "$__RootfsDir/tmp/download/buildtools.zip" -d "$__RootfsDir"

    # Cleaning up temporary files
    echo "Cleaning up temporary files"
    popd
    rm -rf "$__RootfsDir/tmp"
elif [[ -n "$__CodeName" ]]; then
    if [[ "$__SkipEmulation" == "1" ]]; then
        if [[ -z "$AR" ]]; then
            if command -v ar &>/dev/null; then
                AR="$(command -v ar)"
            elif command -v llvm-ar &>/dev/null; then
                AR="$(command -v llvm-ar)"
            else
                echo "Unable to find ar or llvm-ar on PATH, add them to PATH or set AR environment variable pointing to the available AR tool"
                exit 1
            fi
        fi

        # shellcheck disable=SC2086
        suites="$__CodeName $__DebianSuites $(echo $__UbuntuSuites | xargs -n 1 | xargs -I {} echo -n "$__CodeName-{} ")"

        PYTHON=${PYTHON_EXECUTABLE:-python3}

        # shellcheck disable=SC2086,SC2046
        echo running "$PYTHON" "$__CrossDir/install-debs.py" --arch "$__UbuntuArch" --rootfsdir "$__RootfsDir" --artool "$AR" \
            $(echo $suites | xargs -n 1 | xargs -I {} echo -n "--suite {} ") \
            $__UbuntuPackages

        # shellcheck disable=SC2086,SC2046
        "$PYTHON" "$__CrossDir/install-debs.py" --arch "$__UbuntuArch" --mirror "$__UbuntuRepo" --rootfsdir "$__RootfsDir" --artool "$AR" \
            $(echo $suites | xargs -n 1 | xargs -I {} echo -n "--suite {} ") \
            $__UbuntuPackages

        exit 0
    fi

    __UpdateOptions=
    if [[ "$__SkipSigCheck" == "0" ]]; then
        __Keyring="$__Keyring --force-check-gpg"
    else
        __Keyring=
        __UpdateOptions="--allow-unauthenticated --allow-insecure-repositories"
    fi

    # shellcheck disable=SC2086
    echo running debootstrap "--variant=minbase" $__Keyring --arch "$__UbuntuArch" "$__CodeName" "$__RootfsDir" "$__UbuntuRepo"

    # shellcheck disable=SC2086
    if ! debootstrap "--variant=minbase" $__Keyring --arch "$__UbuntuArch" "$__CodeName" "$__RootfsDir" "$__UbuntuRepo"; then
        echo "debootstrap failed! dumping debootstrap.log"
        cat "$__RootfsDir/debootstrap/debootstrap.log"
        exit 1
    fi

    rm -rf "$__RootfsDir"/etc/apt/*.{sources,list} "$__RootfsDir"/etc/apt/sources.list.d
    mkdir -p "$__RootfsDir/etc/apt/sources.list.d/"

    # shellcheck disable=SC2086
    cat > "$__RootfsDir/etc/apt/sources.list.d/$__CodeName.sources" <<EOF
Types: deb
URIs: $__UbuntuRepo
Suites: $__CodeName $__DebianSuites $(echo $__UbuntuSuites | xargs -n 1 | xargs -I {} echo -n "$__CodeName-{} ")
Components: main universe
Signed-By: $__KeyringFile
EOF

    # shellcheck disable=SC2086
    chroot "$__RootfsDir" apt-get update $__UpdateOptions
    chroot "$__RootfsDir" apt-get -f -y install
    # shellcheck disable=SC2086
    chroot "$__RootfsDir" apt-get -y install $__UbuntuPackages
    chroot "$__RootfsDir" symlinks -cr /usr
    chroot "$__RootfsDir" apt-get clean

    if [[ "$__SkipUnmount" == "0" ]]; then
        umount "$__RootfsDir"/* || true
    fi

    if [[ "$__BuildArch" == "armel" && "$__CodeName" == "jessie" ]]; then
        pushd "$__RootfsDir"
        patch -p1 < "$__CrossDir/$__BuildArch/armel.jessie.patch"
        popd
    fi
elif [[ "$__Tizen" == "tizen" ]]; then
    ROOTFS_DIR="$__RootfsDir" "$__CrossDir/tizen-build-rootfs.sh" "$__BuildArch"
else
    echo "Unsupported target platform."
    usage
fi
