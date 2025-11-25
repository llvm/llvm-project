KDIR ?= /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)

obj-m := dsmil_avx512_enabler.o

.PHONY: all clean install

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

install: all
	install -D -m 0644 dsmil_avx512_enabler.ko /lib/modules/$(shell uname -r)/extra/dsmil_avx512_enabler.ko
	depmod -a
