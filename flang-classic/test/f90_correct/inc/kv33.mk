#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test kv33  ########


kv33: run
	

build:  $(SRC)/kv33.f
	-$(RM) kv33.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/kv33.f -o kv33.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) kv33.$(OBJX) check.$(OBJX) $(LIBS) -o kv33.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test kv33
	kv33.$(EXESUFFIX)

verify: ;

kv33.run: run

