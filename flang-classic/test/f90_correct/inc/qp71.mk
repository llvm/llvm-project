#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test binttoq  ########


qp71: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp71.f08 fcheck.$(OBJX)
	-$(RM) qp71.$(EXESUFFIX) qp71.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp71.f08 -o qp71.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp71.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp71.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp71
	qp71.$(EXESUFFIX)

verify: ;

qp71.run: run

