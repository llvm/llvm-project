#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test kv10  ########


kv10: run
	

build:  $(SRC)/kv10.f
	-$(RM) kv10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/kv10.f -o kv10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) kv10.$(OBJX) check.$(OBJX) $(LIBS) -o kv10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test kv10
	kv10.$(EXESUFFIX)

verify: ;

kv10.run: run

