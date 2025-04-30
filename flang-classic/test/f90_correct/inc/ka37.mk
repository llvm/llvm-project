#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ka37  ########


ka37: run
	

build:  $(SRC)/ka37.f
	-$(RM) ka37.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ka37.f -o ka37.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ka37.$(OBJX) check.$(OBJX) $(LIBS) -o ka37.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ka37
	ka37.$(EXESUFFIX)

verify: ;

ka37.run: run

