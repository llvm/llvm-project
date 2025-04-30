#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test kv00  ########


kv00: run
	

build:  $(SRC)/kv00.f
	-$(RM) kv00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/kv00.f -o kv00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) kv00.$(OBJX) check.$(OBJX) $(LIBS) -o kv00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test kv00
	kv00.$(EXESUFFIX)

verify: ;

kv00.run: run

