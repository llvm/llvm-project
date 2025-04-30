#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test kv12  ########


kv12: run
	

build:  $(SRC)/kv12.f
	-$(RM) kv12.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/kv12.f -o kv12.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) kv12.$(OBJX) check.$(OBJX) $(LIBS) -o kv12.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test kv12
	kv12.$(EXESUFFIX)

verify: ;

kv12.run: run

