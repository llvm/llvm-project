#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test kv14  ########


kv14: run
	

build:  $(SRC)/kv14.f
	-$(RM) kv14.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/kv14.f -o kv14.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) kv14.$(OBJX) check.$(OBJX) $(LIBS) -o kv14.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test kv14
	kv14.$(EXESUFFIX)

verify: ;

kv14.run: run

