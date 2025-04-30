
! KGEN-generated Fortran source file
!
! Filename    : parallel_mod.F90
! Generated at: 2015-04-12 19:17:34
! KGEN version: 0.4.9



    MODULE parallel_mod
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        ! ---------------------------
        ! ---------------------------
        IMPLICIT NONE
        PUBLIC
        !
        ! Copyright (C) 2003-2011 Intel Corporation.  All Rights Reserved.
        !
        ! The source code contained or described herein and all documents
        ! related to the source code ("Material") are owned by Intel Corporation
        ! or its suppliers or licensors.  Title to the Material remains with
        ! Intel Corporation or its suppliers and licensors.  The Material is
        ! protected by worldwide copyright and trade secret laws and treaty
        ! provisions.  No part of the Material may be used, copied, reproduced,
        ! modified, published, uploaded, posted, transmitted, distributed, or
        ! disclosed in any way without Intel's prior express written permission.
        !
        ! No license under any patent, copyright, trade secret or other
        ! intellectual property right is granted to or conferred upon you by
        ! disclosure or delivery of the Materials, either expressly, by
        ! implication, inducement, estoppel or otherwise.  Any license under
        ! such intellectual property rights must be express and approved by
        ! Intel in writing.
        !      /* -*- Mode: Fortran; -*- */
        !
        !      (C) 2001 by Argonne National Laboratory.
        !
        !                                 MPICH2 COPYRIGHT
        !
        ! The following is a notice of limited availability of the code, and disclaimer
        ! which must be included in the prologue of the code and in all source listings
        ! of the code.
        !
        ! Copyright Notice
        !  + 2002 University of Chicago
        !
        ! Permission is hereby granted to use, reproduce, prepare derivative works, and
        ! to redistribute to others.  This software was authored by:
        !
        ! Argonne National Laboratory Group
        ! W. Gropp: (630) 252-4318; FAX: (630) 252-5986; e-mail: gropp@mcs.anl.gov
        ! E. Lusk:  (630) 252-7852; FAX: (630) 252-5986; e-mail: lusk@mcs.anl.gov
        ! Mathematics and Computer Science Division
        ! Argonne National Laboratory, Argonne IL 60439
        !
        !
        !                             GOVERNMENT LICENSE
        !
        ! Portions of this material resulted from work developed under a U.S.
        ! Government Contract and are subject to the following license: the Government
        ! is granted for itself and others acting on its behalf a paid-up, nonexclusive,
        ! irrevocable worldwide license in this computer software to reproduce, prepare
        ! derivative works, and perform publicly and display publicly.
        !
        !                                 DISCLAIMER
        !
        ! This computer code material was prepared, in part, as an account of work
        ! sponsored by an agency of the United States Government.  Neither the United
        ! States, nor the University of Chicago, nor any of their employees, makes any
        ! warranty express or implied, or assumes any legal liability or responsibility
        ! for the accuracy, completeness, or usefulness of any information, apparatus,
        ! product, or process disclosed, or represents that its use would not infringe
        ! privately owned rights.
        !
        ! Portions of this code were written by Microsoft. Those portions are
        ! Copyright (c) 2007 Microsoft Corporation. Microsoft grants permission to
        ! use, reproduce, prepare derivative works, and to redistribute to
        ! others. The code is licensed "as is." The User bears the risk of using
        ! it. Microsoft gives no express warranties, guarantees or
        ! conditions. To the extent permitted by law, Microsoft excludes the
        ! implied warranties of merchantability, fitness for a particular
        ! purpose and non-infringement.
        !
        !
        !
        !
        !
        !      DO NOT EDIT
        !      This file created by buildiface
        !
        !S-JMD integer,      public, allocatable :: recvcount(:),displs(:)
        ! ==================================================
        ! Define type parallel_t for distributed memory info
        ! ==================================================
        ! parallel structure for distributed memory programming
        ! ===================================================
        ! Module Interfaces
        ! ===================================================

        PUBLIC abortmp
        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables
        ! ================================================
        !   copy_par: copy constructor for parallel_t type
        !
        !
        !   Overload assignment operator for parallel_t
        ! ================================================

        ! ================================================
        !  initmp:
        !  Initializes the parallel (message passing)
        !  environment, returns a parallel_t structure..
        ! ================================================

        ! =========================================================
        ! abortmp:
        !
        ! Tries to abort the parallel (message passing) environment
        ! and prints a message
        ! =========================================================

        SUBROUTINE abortmp(string)
            CHARACTER(LEN=*) :: string
            !kgen_excluded CALL endrun(string)
        END SUBROUTINE abortmp
        ! =========================================================
        ! haltmp:
        !
        !> stops the parallel (message passing) environment
        !! and prints a message.
        !
        !> Print the message and call MPI_finalize.
        !! @param[in] string The message to be printed.
        ! =========================================================

        ! =========================================================
        ! split:
        !
        ! splits the message passing world into components
        ! and returns a new parallel structure for the
        ! component resident at this process, i.e. lcl_component
        ! =========================================================

        ! =========================================================
        ! connect:
        !
        ! connects this MPI component to all others by constructing
        ! intercommunicator array and storing it in the local parallel
        ! structure lcl_par. Connect assumes you have called split
        ! to create the lcl_par structure.
        !
        ! =========================================================

        ! =====================================
        ! syncmp:
        !
        ! sychronize message passing domains
        !
        ! =====================================

        ! =============================================
        ! pmin_1d:
        ! 1D version of the parallel MIN
        ! =============================================

        ! =============================================
        ! pmax_1d:
        ! 1D version of the parallel MAX
        ! =============================================

        ! =============================================
        ! psum_1d:
        ! 1D version of the parallel MAX
        ! =============================================

    END MODULE parallel_mod
