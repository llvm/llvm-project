#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qp35  ########


qp35: run


build:  $(SRC)/qp35.f08
	-$(RM) qp35.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp35.f08 -o qp35.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp35.$(OBJX) check.$(OBJX) $(LIBS) -o qp35.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp35
	qp35.$(EXESUFFIX)

verify: ;
