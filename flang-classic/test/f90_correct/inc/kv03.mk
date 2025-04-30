#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test kv03  ########


kv03: run
	

build:  $(SRC)/kv03.f
	-$(RM) kv03.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/kv03.f -o kv03.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) kv03.$(OBJX) check.$(OBJX) $(LIBS) -o kv03.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test kv03
	kv03.$(EXESUFFIX)

verify: ;

kv03.run: run

