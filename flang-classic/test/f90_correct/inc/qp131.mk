#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test qpow ########


qp131: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp131.f08 fcheck.$(OBJX)
	-$(RM) qp131.$(EXESUFFIX) qp131.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp131.f08 -o qp131.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp131.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp131.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp131
	qp131.$(EXESUFFIX)

verify: ;

qp131.run: run

