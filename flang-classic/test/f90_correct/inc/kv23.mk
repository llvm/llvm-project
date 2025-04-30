#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test kv23  ########


kv23: run
	

build:  $(SRC)/kv23.f
	-$(RM) kv23.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/kv23.f -o kv23.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) kv23.$(OBJX) check.$(OBJX) $(LIBS) -o kv23.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test kv23
	kv23.$(EXESUFFIX)

verify: ;

kv23.run: run

